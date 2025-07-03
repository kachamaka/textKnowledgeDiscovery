import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from timm.loss.asymmetric_loss import AsymmetricLossMultiLabel
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import nltk
import random
from nltk.corpus import wordnet

nltk.data.path.append("/home/kachamaka/nltk_data")
nltk.download('wordnet', download_dir="/home/kachamaka/nltk_data")
nltk.download('omw-1.4', download_dir="/home/kachamaka/nltk_data")

PERSUASION_TECHNIQUES = ['presenting irrelevant data (red herring)', "misrepresentation of someone's position (straw man)", 'whataboutism', 
                         'causal oversimplification', 'obfuscation, intentional vagueness, confusion', 'appeal to authority', 
                         'black-and-white fallacy/dictatorship', 'name calling/labeling', 'loaded language', 'exaggeration/minimisation', 
                         'flag-waving', 'doubt', 'appeal to fear/prejudice', 'slogans', 'thought-terminating cliché', 'bandwagon', 
                         'reductio ad hitlerum', 'repetition', 'smears', 'glittering generalities (virtue)', 'transfer', 'appeal to (strong) emotions']

HIERARCHY = {
    "name calling/labeling": ["ad hominem"],
    "doubt": ["ad hominem"],
    "smears": ["ad hominem"],
    "reductio ad hitlerum": ["ad hominem"],
    "whataboutism": ["ad hominem", "distraction"],

    "ad hominem": ["ethos"],

    "bandwagon": ["ethos", "justification"],
    "appeal to authority": ["ethos", "justification"],
    "glittering generalities (virtue)": ["ethos"],
    "transfer": ["ethos", "pathos"],

    "appeal to (strong) emotions": ["pathos"],
    "exaggeration/minimisation": ["pathos"],
    "loaded language": ["pathos"],
    "flag-waving": ["pathos", "justification"],
    "appeal to fear/prejudice": ["pathos", "justification"],
    "slogans": ["justification"],

    "justification": ["logos"],
    "repetition": ["logos"],
    "obfuscation, intentional vagueness, confusion": ["logos"],

    "reasoning": ["logos"],
    "distraction": ["reasoning"],
    "simplification": ["reasoning"],

    "misrepresentation of someone's position (straw man)": ["distraction"],
    "presenting irrelevant data (red herring)": ["distraction"],

    "causal oversimplification": ["simplification"],
    "black-and-white fallacy/dictatorship": ["simplification"],
    "thought-terminating cliché": ["simplification"],

    "ethos": ["persuasion"],
    "pathos": ["persuasion"],
    "logos": ["persuasion"]
}

def get_ancestors(label, ancestors=None):
    if ancestors is None:
        ancestors = set()
    ancestors.add(label)
    if label in HIERARCHY:
        for parent in HIERARCHY[label]:
            if parent not in ancestors:
                get_ancestors(parent, ancestors)
    return ancestors

def expand_labels(labels):
    expanded = set()
    for label in labels:
        expanded |= get_ancestors(label)

    return expanded

def labels_to_vector(labels):
    labels = [l.lower() for l in labels] # Normalize labels to lowercase

    one_hot_vector = torch.zeros(len(PERSUASION_TECHNIQUES), dtype=torch.float32)
    for label in labels:
        idx = PERSUASION_TECHNIQUES.index(label)
        one_hot_vector[idx] = 1.0
    return one_hot_vector

# # one_hot_vector = batch_size x num_classes
# def vector_to_labels(one_hot_vector):
#     batch_labels = []
#     for _, batch in enumerate(one_hot_vector):
#         labels = []
#         for idx, value in enumerate(batch):
#             if value > 0:
#                 labels.append(PERSUASION_TECHNIQUES[idx])
#         batch_labels.append(labels)

#     return batch_labels


# one_hot_vector = batch_size x num_classes
def vector_to_labels(one_hot_vector):
    # print("one_hot_vector:", one_hot_vector)
    labels = []
    for idx, value in enumerate(one_hot_vector):
        if value > 0:
            labels.append(PERSUASION_TECHNIQUES[idx])

    return labels

# logits = batch_size x num_classes
def logits_to_one_hot(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    one_hot_vector = (probs > threshold).int().tolist()
    return one_hot_vector

def hierarchical_f1_score(pred_labels, gold_labels, verbose=False):
    gold_nodes = expand_labels(gold_labels) # green
    pred_nodes = expand_labels(pred_labels) # yellow
    intersection = gold_nodes.intersection(pred_nodes) # orange

    precision = len(intersection) / len(pred_nodes) if pred_nodes else 0.0
    recall = len(intersection) / len(gold_nodes) if gold_nodes else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    if verbose:
            print(f"Gold Labels (Green): {gold_labels}")
            print(f"Predicted Labels (Yellow): {pred_labels}")
            print(f"Expanded Pred Nodes (Yellow): {pred_nodes}")
            print(f"Expanded Gold Nodes (Green): {gold_nodes}")
            print(f"Intersection (Orange): {intersection}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("=" * 60)
            # time.sleep(5)
    return f1

def f1_scores(targets, predictions):
    # macro_f1 = f1_score(targets, predictions, average='macro')
    # micro_f1 = f1_score(targets, predictions, average='micro')
    # return macro_f1, micro_f1
    # TODO: experiment with different scores
    all_f1_scores = []
    for pred, target in zip(predictions, targets):
        pred = vector_to_labels(pred)
        target = vector_to_labels(target)
        f1 = hierarchical_f1_score(pred, target)
        all_f1_scores.append(f1)

    macro_f1 = np.mean(all_f1_scores)
    return macro_f1, 0

def plot_training_metrics(train_history):
    plt.figure(figsize=(10, 12))

    # Plot 1: Loss
    plt.subplot(2, 1, 1)
    plt.plot(train_history['train_losses'], label='Train Loss', color='blue')
    plt.plot(train_history['val_losses'], label='Validation Loss', color='red')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Plot 2: Macro F1
    plt.subplot(2, 1, 2)
    plt.plot(train_history['train_macro_f1_scores'], label='Train Macro F1', color='blue')
    plt.plot(train_history['val_macro_f1_scores'], label='Validation Macro F1', color='red')
    plt.title('Macro F1 Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Macro F1 Score')
    plt.legend()
    plt.grid()

    # Plot 3: Micro F1
    # plt.subplot(3, 1, 3)
    # plt.plot(train_history['train_micro_f1_scores'], label='Train Micro F1', color='blue')
    # plt.plot(train_history['val_micro_f1_scores'], label='Validation Micro F1', color='red')
    # plt.title('Micro F1 Score over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Micro F1 Score')
    # plt.legend()
    # plt.grid()

    plt.tight_layout()
    plt.show()

def extract_captions():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    import json
    import pandas as pd
    import os
    # caption_model = "microsoft/Florence-2-large-ft"
    caption_model = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(caption_model)
    model = BlipForConditionalGeneration.from_pretrained(caption_model)
    model.eval()

    # Preprocess

    train_captions = []
    train_df = pd.read_json("./labels/train.json")
    for _, row in train_df.iterrows():
        img_name = row['image']
        image_path = os.path.join("data", "train", img_name)
        image = Image.open(image_path).convert('RGB')

        # Generate caption
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs)

        train_caption = processor.decode(output[0], skip_special_tokens=True)
        print(len(train_captions), "/", len(train_df))
        print(f"Generating caption for image: {img_name}")
        # print(f"Generated caption: {train_caption}")
        train_captions.append({
            "id": row['id'],
            "text": train_caption,
            "image": img_name,
            "labels": row['labels'],
        })
    
    with open("./labels/train_captions1.json", "w") as f:
        json.dump(train_captions, f, indent=4)

def create_label_to_index_mapping():
    """Create mapping from all unique labels (leaves + internal nodes) to indices."""
    all_labels = set()
    for child, parents in HIERARCHY.items():
        all_labels.add(child)
        all_labels.update(parents)

    # Ensure root is included
    all_labels.add('persuasion')

    # Sort for consistent indexing
    return {label: idx for idx, label in enumerate(sorted(all_labels))}

def create_label_index_mappings():
    """Return both label_to_index and index_to_label mappings for all nodes."""
    all_labels = set()
    for child, parents in HIERARCHY.items():
        all_labels.add(child)
        all_labels.update(parents)

    all_labels.add('persuasion')

    sorted_labels = sorted(all_labels)
    label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    return label_to_index, index_to_label


class HierarchicalBCELoss(nn.Module):
    """
    Hierarchical BCE Loss that enforces ancestor consistency.
    If a child technique is predicted, its ancestors should also be predicted.
    """
    def __init__(self, pos_weight=None, alpha=0.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.alpha = alpha  # Balance between flat and hierarchical loss
        
    def forward(self, logits, targets):
        losses = []
        for logit, target in zip(logits, targets):
            flat_loss = F.binary_cross_entropy_with_logits(logit, target, pos_weight=self.pos_weight if self.training else None)
            
            one_hot_prediction = logits_to_one_hot(logit)

            predicted_labels = vector_to_labels(one_hot_prediction)  # yellow
            target_labels = vector_to_labels(target)  # green
            
            pred_nodes = expand_labels(predicted_labels) # yellow
            gold_nodes = expand_labels(target_labels) # green

            mapping = create_label_to_index_mapping()
            
            prediction_nodes = torch.zeros(len(mapping))
            target_nodes = torch.zeros(len(mapping))
            for label in pred_nodes:
                prediction_nodes[mapping[label]] = 1.0
            for label in gold_nodes:
                target_nodes[mapping[label]] = 1.0

            # print(f"Prediction Nodes: {prediction_nodes}", prediction_nodes.shape)
            # print(f"Target Nodes: {target_nodes}", target_nodes.shape)
            # intersection = gold_nodes.intersection(pred_nodes) # orange

            hierarchical_loss = F.binary_cross_entropy(prediction_nodes, target_nodes)
            # hierarchical_loss = F.mse_loss(prediction_nodes, target_nodes)
        
            # Combine losses
            total_loss = self.alpha * flat_loss + (1 - self.alpha) * hierarchical_loss
            losses.append(total_loss)
        
        losses = torch.stack(losses).mean()  # Average over batch
        return losses
