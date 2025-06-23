import os
import math
import torch
import json
import matplotlib.pyplot as plt
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import nn
from torch.nn import init
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np
from collections import Counter
import timm
from timm.loss.asymmetric_loss import AsymmetricLossMultiLabel
from torchvision.transforms import RandAugment
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

TRAIN_TRANSFORMATIONS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    RandAugment(num_ops=2, magnitude=5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMATIONS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_concept_list(cui_csv_path):
    """Load the concept list from CUI CSV file"""
    df = pd.read_csv(cui_csv_path)
    return df['CUI'].tolist()

class ConceptDataset(Dataset):
    def __init__(self, data_split_dir, csv_file, concept_list, transform=None):
        self.data_split_dir = data_split_dir
        self.transform = transform
        self.samples = []
        self.concept_list = concept_list
        self.class_counter = Counter()

        self.concept_to_idx = {cid: idx for idx, cid in enumerate(self.concept_list)}

        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            img_name = row['ID']
            concept_ids = row['CUIs']
            image_path = os.path.join(self.data_split_dir, img_name + ".jpg")

            if not os.path.exists(image_path):
                continue

            multi_hot = torch.zeros(len(self.concept_list), dtype=torch.float32)

            for cid in str(concept_ids).split(";"):
                cid = cid.strip()
                if cid in self.concept_to_idx:
                    idx = self.concept_to_idx[cid]
                    multi_hot[idx] = 1
                    self.class_counter[cid] += 1

            self.samples.append((image_path, multi_hot))
            # return # one sample for testing

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, labels

    def print_class_distribution(self):
        print(f"\nTotal concepts: {len(self.concept_list)}")
        print(f"Total samples: {len(self.samples)}")

        # Analyze label distribution
        all_labels = []
        for _, labels in self.samples:
            all_labels.append(labels.numpy())

        all_labels = np.array(all_labels)

        # Per-class statistics
        samples_per_class = all_labels.sum(axis=0)
        positive_classes = (samples_per_class > 0).sum()
        print(f"Classes with positive samples: {positive_classes}/{len(self.concept_list)}")
        print(f"Samples per class - Mean (of positive classes): {samples_per_class[samples_per_class > 0].mean():.2f}")

        # Show distribution of class frequencies
        rare_classes = ((samples_per_class > 0) & (samples_per_class < 5)).sum()
        medium_classes = ((samples_per_class >= 5) & (samples_per_class < 50)).sum()
        common_classes = (samples_per_class >= 50).sum()

        print(f"Class frequency distribution:")
        print(f"  Rare classes (<5 samples): {rare_classes}")
        print(f"  Medium classes (5-49 samples): {medium_classes}")
        print(f"  Common classes (≥50 samples): {common_classes}")

def compute_sample_weights(dataset, class_weights):
    sample_weights = []

    for _, label in dataset.samples:
        label = label.to(DEVICE)
        # Multiply each multi-hot label vector by class_weights and take the sum
        weight = (label * class_weights).sum().item()
        sample_weights.append(weight)

    return sample_weights

def compute_class_weights(dataset, concept_list, samples_per_concept, strategy='sqrt_inv_freq'):
    """Compute class weights using several robust strategies."""
    total_samples = len(dataset)
    num_concepts = len(concept_list)
    class_weights = torch.ones(num_concepts, device=DEVICE)

    for i, concept_id in enumerate(concept_list):
        pos_count = max(samples_per_concept.get(concept_id, 0), 1)  # Avoid divide-by-zero
        if strategy == 'sqrt_inv_freq':
            class_weights[i] = math.sqrt(total_samples / pos_count)
        elif strategy == 'balanced':
            class_weights[i] = total_samples / (2.0 * pos_count)
        elif strategy == 'effective_number':
            beta = 1.0 - 1.0 / total_samples
            effective_num = 1.0 - math.pow(beta, pos_count)
            class_weights[i] = (1.0 - beta) / (effective_num + 1e-8)
        elif strategy == 'focal_inspired':
            pos_ratio = pos_count / total_samples
            class_weights[i] = 1.0 / (pos_ratio + 1e-7)
        elif strategy == 'log_inv_freq':
            class_weights[i] = math.log(1 + total_samples / (pos_count + 1))
        elif strategy == 'hybrid':
            inv = total_samples / (pos_count + 1e-7)
            beta = 0.9999
            eff_num = (1 - beta) / (1 - beta**pos_count + 1e-8)
            class_weights[i] = math.sqrt(inv * eff_num)
        else:
            class_weights[i] = 1.0

    class_weights = torch.clamp(class_weights, min=0.5, max=30.0)
    class_weights = class_weights * (len(concept_list) / class_weights.sum())

    return class_weights

class FocalAsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=4, clip=0.05, alpha=0.25, focal_weight=0.5, class_weights=None, focal_gamma=2.0):
        super().__init__()
        self.asymmetric_loss = AsymmetricLossMultiLabel(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)
        self.alpha = alpha
        self.focal_weight = focal_weight
        self.class_weights = class_weights 
        self.focal_gamma = focal_gamma

    def focal_loss(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            pos_weight=self.class_weights if self.training else None, # dont use class weights for validation
            reduction='none'
        )

        pt = torch.exp(-BCE_loss.clamp(min=1e-6))  # no max
        # pt = torch.exp(-BCE_loss.clamp(min=1e-6, max=10))
        # pt = torch.exp(-BCE_loss.clamp(min=1e-6, max=50))
        focal_mod = (1.0 - pt).pow(self.focal_gamma)
        focal_loss = self.alpha * focal_mod * BCE_loss

        return focal_loss.mean()

    def forward(self, inputs, targets):
        asymmetric_loss = self.asymmetric_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)

        return (1.0 - self.focal_weight) * asymmetric_loss + self.focal_weight * focal_loss

class ConceptDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(ConceptDetectionModel, self).__init__()

        # self.model = timm.create_model('coat_lite_medium', pretrained=True)
        # self.model = timm.create_model('mobilenetv3_large_100', pretrained=True)
        # self.model = timm.create_model('resnet50', pretrained=True)
        # self.model = timm.create_model('vit_base_resnet50_224_in21k', pretrained=True, num_classes=num_classes)
        # self.model = timm.create_model('vit_large_patch16_224_in21k', pretrained=True, num_classes=num_classes)
        # self.model = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True, num_classes=num_classes)

        # self.model = timm.create_model('convnext_large_in22ft1k', pretrained=True, num_classes=num_classes)
        # self.model = timm.create_model('convnext_base', pretrained=True, num_classes=num_classes)
        self.model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=num_classes)
        # self.model = timm.create_model('convnext_tiny.fb_in22k_ft_in1k_384', pretrained=True, num_classes=num_classes)
        # self.model = timm.create_model('convnext_tiny.fb_in22k_ft_in1k_384', pretrained=True, num_classes=num_classes)
        # self.model = timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=True, num_classes=num_classes)
        # self.model = timm.create_model('convnext_large.fb_in22k_ft_in1k', pretrained=True, num_classes=num_classes)
        
        # self.model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True)
        # self.model = timm.create_model('beit_base_patch16_224.in22k_ft_in22k', pretrained=True, num_classes=num_classes)

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if "stages.3" in name or "head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # for name, param in self.model.named_parameters():
        #     if name in ["conv_head"]:
        #         param.requires_grad = True

        # for param in self.model.head.fc.parameters():
        #     param.requires_grad = True

        # in_features = self.model.classifier.in_features
        # self.model.classifier = nn.Linear(in_features, num_classes)
        
        # in_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(in_features, num_classes)
        
        # in_features = self.model.head.in_features
        # self.model.head = nn.Linear(in_features, num_classes)

        in_features = self.model.head.fc.in_features
        # self.model.head.fc = nn.Linear(in_features, num_classes)
        self.model.head.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        

    def forward(self, x):
        return self.model(x)

def calculate_metrics(targets, predictions, thresholds=0.5):

    pred_binary = (predictions >= thresholds).float()

    # Convert to numpy for sklearn metrics
    targets_np = targets.cpu().numpy()
    pred_binary_np = pred_binary.cpu().numpy()

    # Calculate metrics
    # zero_division=0 ensures that F1 is 0 for classes where there are no true positives
    # and the model doesn't predict any, or for classes with no true positives at all.
    macro_f1 = f1_score(targets_np, pred_binary_np, average='macro', zero_division=0)
    micro_f1 = f1_score(targets_np, pred_binary_np, average='micro', zero_division=0)
    weighted_f1 = f1_score(targets_np, pred_binary_np, average='weighted', zero_division=0)

    # Per-class F1 scores
    _, _, per_class_f1, _ = precision_recall_fscore_support(
        targets_np, pred_binary_np, average=None, zero_division=0
    )

    return macro_f1, micro_f1, weighted_f1, per_class_f1

def optimize_thresholds_for_micro_f1(y_true, y_probs, n_thresholds=10):
    """Optimize thresholds specifically for macro F1"""
    y_true_np = y_true.cpu().numpy()
    y_probs_np = y_probs.cpu().numpy()
    
    n_classes = y_true_np.shape[1]
    optimal_thresholds = np.full(n_classes, 0.5)
    
    # Use more threshold candidates
    threshold_candidates = np.linspace(0.05, 0.95, n_thresholds)
    
    for class_idx in range(n_classes):
        best_f1 = 0
        best_threshold = 0.5
        
        # Skip classes with no positive samples
        if y_true_np[:, class_idx].sum() == 0:
            continue
            
        for threshold in threshold_candidates:
            y_pred_class = (y_probs_np[:, class_idx] >= threshold).astype(int)
            f1 = f1_score(y_true_np[:, class_idx], y_pred_class, average='macro', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds[class_idx] = best_threshold
    
    return optimal_thresholds

def train_model(dataloader_train, dataloader_validation, model, optimizer, num_epochs, scheduler, class_weights=None, threshold=0.5, save_path=None):
    model_name = getattr(model.model, 'default_cfg', {}).get('architecture', 'unnamed_model')
    model_dir = os.path.join("models", model_name)
    os.makedirs(model_dir, exist_ok=True)

    criterion = FocalAsymmetricLoss(gamma_neg=3, gamma_pos=1, clip=0.05, alpha=0.5, focal_weight=0.7, class_weights=class_weights).to(DEVICE)

    train_losses = []
    train_macro_f1_scores = []
    train_micro_f1_scores = []
    train_weighted_f1_scores = []
    val_losses = []
    val_macro_f1_scores = []
    val_micro_f1_scores = []
    val_weighted_f1_scores = []
    best_macro_val_f1 = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        all_train_targets = []
        all_train_predictions = []

        train_bar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (images, targets) in enumerate(train_bar):
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch + batch_idx / len(dataloader_train))  


            train_loss += loss.item()

            # Collect predictions for metrics
            with torch.no_grad():
                probs = torch.sigmoid(predictions)
                all_train_targets.append(targets)
                all_train_predictions.append(probs)

            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        all_train_targets = torch.cat(all_train_targets, dim=0)
        all_train_predictions = torch.cat(all_train_predictions, dim=0)

        # Calculate training metrics using default 0.5 threshold for general understanding
        train_macro_f1, train_micro_f1, train_weighted_f1, train_per_class_f1 = calculate_metrics(all_train_targets, all_train_predictions, thresholds=threshold)
        train_macro_f1_scores.append(train_macro_f1)
        train_micro_f1_scores.append(train_micro_f1)
        train_weighted_f1_scores.append(train_weighted_f1)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_targets = []
        all_val_predictions = []

        with torch.no_grad():
            val_bar = tqdm(dataloader_validation, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, targets in val_bar:
                images, targets = images.to(DEVICE), targets.to(DEVICE)

                predictions = model(images)
                loss = criterion(predictions, targets)
                val_loss += loss.item()

                probs = torch.sigmoid(predictions)
                all_val_targets.append(targets)
                all_val_predictions.append(probs)

                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        all_val_targets = torch.cat(all_val_targets, dim=0)
        all_val_predictions = torch.cat(all_val_predictions, dim=0)

        optimal_thresholds = optimize_thresholds_for_micro_f1(all_val_targets, all_val_predictions)
        optimal_thresholds = torch.tensor(optimal_thresholds, device=all_val_predictions.device, dtype=all_val_predictions.dtype)

        thresholds_path = os.path.join(model_dir, f"epoch_{epoch+1}_thresholds.json")
        with open(thresholds_path, "w") as f:
            json.dump(optimal_thresholds.tolist(), f)


        val_macro_f1, val_micro_f1, val_weighted_f1, val_per_class_f1 = calculate_metrics(all_val_targets, all_val_predictions, thresholds=optimal_thresholds)
        # val_macro_f1, val_micro_f1, val_weighted_f1, val_per_class_f1 = calculate_metrics(all_val_targets, all_val_predictions, thresholds=threshold)
        val_macro_f1_scores.append(val_macro_f1)
        val_micro_f1_scores.append(val_micro_f1)
        val_weighted_f1_scores.append(val_weighted_f1)

        avg_train_loss = train_loss / len(dataloader_train)
        train_losses.append(avg_train_loss)
        avg_val_loss = val_loss / len(dataloader_validation)
        val_losses.append(avg_val_loss)

        train_bar.write(f"Train Loss: {avg_train_loss:.4f}, Macro F1: {train_macro_f1:.4f}, Micro F1: {train_micro_f1:.4f}, Weighted F1: {train_weighted_f1:.4f}")
        val_bar.write(f"Val Loss: {avg_val_loss:.4f}, Macro F1: {val_macro_f1:.4f}, Micro F1: {val_micro_f1:.4f}, Weighted F1: {val_weighted_f1:.4f}")

        # -----
        opt_train_macro_f1, opt_train_micro_f1, _, _ = calculate_metrics(all_train_targets, all_train_predictions, thresholds=optimal_thresholds)
        train_bar.write(f"Optimized Macro F1: {train_macro_f1:.4f}, Optimized Micro F1: {train_micro_f1:.4f}")
        # -----

        checkpoint_path = os.path.join(model_dir, f"epoch_{epoch+1}_micro{val_micro_f1:.4f}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_macro_f1': val_macro_f1,
            'val_micro_f1': val_micro_f1,
            'val_weighted_f1': val_weighted_f1,
            'concept_list': dataloader_train.dataset.concept_list,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model based on validation macro F1
        # if val_macro_f1 > best_macro_val_f1:
        #     best_macro_val_f1 = val_macro_f1
        #     if save_path:
        #         torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'best_macro_val_f1': best_macro_val_f1,
        #             'concept_list': dataloader_train.dataset.concept_list,
        #         }, save_path)
        #         print(f"Best model saved with validation macro F1: {best_macro_val_f1:.4f}")

    return {
        'train_losses': train_losses,
        'train_macro_f1_scores': train_macro_f1_scores,
        'train_micro_f1_scores': train_micro_f1_scores,
        'train_weighted_f1_scores': train_weighted_f1_scores,
        'val_losses': val_losses,
        'val_macro_f1_scores': val_macro_f1_scores,
        'val_micro_f1_scores': val_micro_f1_scores,
        'val_weighted_f1_scores': val_weighted_f1_scores,
    }

def plot_training_metrics(train_history):
    plt.figure(figsize=(18, 12))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(train_history['train_losses'], label='Train Loss')
    plt.plot(train_history['val_losses'], label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Macro F1
    plt.subplot(2, 2, 2)
    plt.plot(train_history['train_macro_f1_scores'], label='Train Macro F1')
    plt.plot(train_history['val_macro_f1_scores'], label='Val Macro F1')
    plt.title('Macro F1 Score per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1 Score')
    plt.legend()
    plt.grid(True)

    # Micro F1
    plt.subplot(2, 2, 3)
    plt.plot(train_history['train_micro_f1_scores'], label='Train Micro F1')
    plt.plot(train_history['val_micro_f1_scores'], label='Val Micro F1')
    plt.title('Micro F1 Score per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Micro F1 Score')
    plt.legend()
    plt.grid(True)

    # Weighted F1
    plt.subplot(2, 2, 4)
    plt.plot(train_history['train_weighted_f1_scores'], label='Train Weighted F1')
    plt.plot(train_history['val_weighted_f1_scores'], label='Val Weighted F1')
    plt.title('Weighted F1 Score per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted F1 Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    data_dir = "data"
    context_dir = "context"
    cui_csv_path = "db/cui_names.csv"
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 0
    threshold = 0.5
    num_epochs = 20

    print(f"Using device: {DEVICE}")

    concept_list = load_concept_list(cui_csv_path)
    print(f"Total concepts in database: {len(concept_list)}")

    train_dataset = ConceptDataset(
        data_split_dir=os.path.join(data_dir, "train"),
        csv_file=os.path.join(context_dir, "train", "train_concepts.csv"),
        concept_list=concept_list,
        transform=TRAIN_TRANSFORMATIONS,
    )

    val_dataset = ConceptDataset(
        data_split_dir=os.path.join(data_dir, "val"),
        csv_file=os.path.join(context_dir, "val", "valid_concepts.csv"),
        concept_list=concept_list,
        transform=VAL_TRANSFORMATIONS,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of concepts: {len(concept_list)}")

    train_dataset.print_class_distribution()

    num_concepts = len(concept_list)

    # class_weights = compute_class_weights(train_dataset, concept_list, train_dataset.class_counter, strategy='effective_number')
    # class_weights = compute_class_weights(train_dataset, concept_list, train_dataset.class_counter, strategy='focal_inspired')
    class_weights = compute_class_weights(train_dataset, concept_list, train_dataset.class_counter, strategy='sqrt_inv_freq')
    # class_weights = compute_class_weights(train_dataset, concept_list, train_dataset.class_counter, strategy='balanced')
    # class_weights = compute_class_weights(train_dataset, concept_list, train_dataset.class_counter, strategy='hybrid')

    sample_weights = compute_sample_weights(train_dataset, class_weights)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ConceptDetectionModel(num_concepts).to(DEVICE)
    print(f"Model created with {num_concepts} output classes")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    print("Starting training...")
    train_history = train_model(
        train_loader,
        val_loader,
        model,
        optimizer,
        num_epochs,
        scheduler,
        class_weights,
        threshold=threshold,
        save_path="concept_detection.pth"
    )
    
    plot_training_metrics(train_history)

    print("Training completed!")

if __name__ == "__main__":
    main()