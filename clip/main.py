import utils

import os
import torch
from torchvision import transforms
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
from collections import Counter
from transformers import CLIPProcessor, CLIPModel

# MODEL = "openai/clip-vit-base-patch32"
MODEL = "openai/clip-vit-base-patch16"

CLIPPROCESSOR = CLIPProcessor.from_pretrained(MODEL)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

TRAIN_TRANSFORMATIONS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandAugment(num_ops=3, magnitude=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIPPROCESSOR.feature_extractor.image_mean, std=CLIPPROCESSOR.feature_extractor.image_std),
])

VAL_TRANSFORMATIONS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIPPROCESSOR.feature_extractor.image_mean, std=CLIPPROCESSOR.feature_extractor.image_std),
])

class PropagandaDataset(Dataset):
    def __init__(self, data_split, labels, transform=None, use_captions=False):
        self.data_split = data_split
        self.transform = transform
        self.samples = []
        self.sample_counter = Counter()     

        df = pd.read_json(labels)

        if use_captions:
            self.captions = pd.read_json(f"./labels/{data_split}_captions.json").to_dict()['text']

            if len(self.captions) != len(df):
                raise ValueError(f"Number of captions ({len(self.captions)}) does not match number of samples ({len(df)}) in {data_split} set.")

        count = 500
        for idx, row in df.iterrows():
            img_name = row['image']
            image_path = os.path.join("data", self.data_split, img_name)
            labels = [x.lower() for x in row['labels']] # Normalize labels
            img_text = row['text'].lower()
            
            for label in set(labels):
                self.sample_counter[label] += 1
                
            labels_one_hot = utils.labels_to_vector(labels)
            self.samples.append((image_path, img_text, labels_one_hot))

            if use_captions:
                caption = self.captions[idx].lower()
                self.samples.append((image_path, caption, labels_one_hot))
                
            # return # One sample for testing
            # if idx >= count:
            #     return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, img_text, labels = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, img_text, labels
    
    def calculate_class_weights(self):
        class_counts = torch.zeros(len(self.sample_counter))
        total_samples = len(self)
        
        for i, (label, count) in enumerate(self.sample_counter.most_common()):
            idx = utils.PERSUASION_TECHNIQUES.index(label)
            class_counts[idx] = count
        
        # Inverse frequency weighting
        class_weights = total_samples / (len(utils.PERSUASION_TECHNIQUES) * class_counts + 1e-6)
        return class_weights

    def print_class_distribution(self):
        total_classes = len(self.sample_counter)
        total_samples = len(self.samples)

        print(f"\nClass Distribution Summary for {self.data_split}:")
        print(f"Total classes               : {total_classes}")
        print(f"Total samples               : {total_samples}")

        print("\nSample count per class:")
        for label, count in self.sample_counter.most_common(5):
            print(f"{label:50s}: {count} samples")
   
class PersuasionTechniquesModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2):
        super(PersuasionTechniquesModel, self).__init__()

        self.model = CLIPModel.from_pretrained(MODEL)
        self.dropout_rate = dropout_rate

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in [
                "visual.transformer.resblocks.10",
                "text_model.encoder.layers.10",
                "visual.transformer.resblocks.11",
                "text_model.encoder.layers.11",
            ]):
                param.requires_grad = True

        hidden_dim = self.model.config.projection_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        image_embeds = outputs.image_embeds  # (batch_size, projection_dim)
        text_embeds = outputs.text_embeds    # (batch_size, projection_dim)

        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        logits = self.classifier(combined_embeds)

        return logits

def collate_fn(batch):
    images, texts, labels = zip(*batch)
    # Process images and texts with CLIP processor
    # inputs = CLIPPROCESSOR(text=list(texts), images=list(images), return_tensors="pt", padding=True, truncation=True)
    # inputs = CLIPPROCESSOR(text=list(texts), images=list(images), return_tensors="pt", padding=True, truncation=True, do_resize=False)
    # processed_images = [custom_image_transform(img) for img in images]
    pixel_values = torch.stack(images)

    # Tokenize text separately
    text_inputs = CLIPPROCESSOR.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)

    labels = torch.stack(labels)
    # return inputs['pixel_values'], inputs['input_ids'], inputs['attention_mask'], labels
    return pixel_values, text_inputs['input_ids'], text_inputs['attention_mask'], labels

def train_model(dataloader_train, dataloader_validation, model, optimizer, num_epochs, threshold=0.5, save_path=None):
    
    class_weights = dataloader_train.dataset.calculate_class_weights().to(DEVICE)
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # TODO: experiment with different alpha
    criterion = utils.HierarchicalBCELoss(pos_weight=class_weights, alpha=0.6).to(DEVICE)

    train_losses = []
    train_macro_f1_scores = []
    train_micro_f1_scores = []
    val_losses = []
    val_macro_f1_scores = []
    val_micro_f1_scores = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        all_train_targets = []
        all_train_predictions = []

        train_bar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for pixel_values, input_ids, attention_mask, labels in train_bar:
            pixel_values, input_ids, attention_mask, labels = pixel_values.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()

            logits = model(pixel_values, input_ids, attention_mask)

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            # Collect predictions for metrics
            with torch.no_grad():
                predictions_one_hot = utils.logits_to_one_hot(logits, threshold=threshold) # num_batches x PREDICTED_CLASSES
                
                all_train_predictions.extend(predictions_one_hot)
                # predictions_labels = vector_to_labels(predictions_one_hot)
                # all_train_predictions.extend(predictions_labels)

                all_train_targets.extend(labels.tolist())
                # target_labels = vector_to_labels(labels)
                # all_train_targets.extend(target_labels)

            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(dataloader_train)
        train_losses.append(avg_train_loss)

        train_macro_f1_score, train_micro_f1_score = utils.f1_scores(all_train_targets, all_train_predictions)
        train_macro_f1_scores.append(train_macro_f1_score)
        train_micro_f1_scores.append(train_micro_f1_score)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_targets = []
        all_val_predictions = []

        with torch.no_grad():
            val_bar = tqdm(dataloader_validation, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for pixel_values, input_ids, attention_mask, labels in val_bar:
                pixel_values, input_ids, attention_mask, labels = pixel_values.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)

                logits = model(pixel_values, input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                predictions_one_hot = utils.logits_to_one_hot(logits, threshold=threshold) # num_batches x PREDICTED_CLASSES
                
                all_val_predictions.extend(predictions_one_hot)
                # predictions_labels = vector_to_labels(predictions_one_hot)
                # all_val_predictions.extend(predictions_labels)

                all_val_targets.extend(labels.tolist())
                # target_labels = vector_to_labels(labels)
                # all_val_targets.extend(target_labels)

                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / len(dataloader_validation)
        val_losses.append(avg_val_loss)

        val_macro_f1_score, val_micro_f1_score = utils.f1_scores(all_val_targets, all_val_predictions)
        val_macro_f1_scores.append(val_macro_f1_score)
        val_micro_f1_scores.append(val_micro_f1_score)

        train_bar.write(f"Train Loss: {avg_train_loss:.4f}, Macro F1: {train_macro_f1_score:.4f}, Micro F1: {train_micro_f1_score:.4f}")
        val_bar.write(f"Val Loss: {avg_val_loss:.4f}, Macro F1: {val_macro_f1_score:.4f}, Micro F1: {val_micro_f1_score:.4f}")

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            model_file = os.path.join(save_path, f'epoch_{epoch+1}.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_macro_f1': train_macro_f1_score,
                'train_micro_f1': train_micro_f1_score,
                'val_macro_f1': val_macro_f1_score,
                'val_micro_f1': val_micro_f1_score,
            }
    
            torch.save(checkpoint, model_file)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_macro_f1_scores': train_macro_f1_scores,
        'train_micro_f1_scores': train_micro_f1_scores,
        'val_macro_f1_scores': val_macro_f1_scores,
        'val_micro_f1_scores': val_micro_f1_scores,
    }

def main():
    batch_size = 8
    learning_rate = 5e-4
    weight_decay = 1e-6
    threshold = 0.5
    num_epochs = 20

    print(f"Using device: {DEVICE}")

    train_dataset = PropagandaDataset(
        data_split="train", 
        labels="./labels/train.json", 
        transform=TRAIN_TRANSFORMATIONS, 
        use_captions=True, 
    )
    
    val_dataset = PropagandaDataset(
        data_split="val", 
        labels="./labels/val.json", transform=VAL_TRANSFORMATIONS
)

    train_dataset.print_class_distribution()
    val_dataset.print_class_distribution()

    num_classes = len(utils.PERSUASION_TECHNIQUES)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = PersuasionTechniquesModel(num_classes).to(DEVICE)
    print(f"Model created with {num_classes} output classes")

    clip_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                clip_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': clip_params, 'lr': learning_rate * 0.1},  # Lower LR for CLIP
        {'params': classifier_params, 'lr': learning_rate}   # Higher LR for classifier
    ], weight_decay=weight_decay)

    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)

    print("Starting training...")
    train_history = train_model(
        train_loader,
        val_loader,
        model,
        optimizer,
        num_epochs,
        threshold=threshold,
        save_path=f"models/{MODEL}"
    )
    
    utils.plot_training_metrics(train_history)

    print("Training completed!")

if __name__ == "__main__":
    main()