import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import nltk
import random
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

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

# one_hot_vector = batch_size x num_classes
def vector_to_labels(one_hot_vector):
    batch_labels = []
    for _, batch in enumerate(one_hot_vector):
        labels = []
        for idx, value in enumerate(batch):
            if value > 0:
                labels.append(PERSUASION_TECHNIQUES[idx])
        batch_labels.append(labels)

    return batch_labels

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
            print(f"Expanded Pred Nodes (Yellow): {pred_nodes}")
            print(f"Expanded Gold Nodes (Green): {gold_nodes}")
            print(f"Intersection (Orange): {intersection}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("=" * 60)
    return f1

def f1_scores(targets, predictions):
    macro_f1 = f1_score(targets, predictions, average='macro')
    micro_f1 = f1_score(targets, predictions, average='micro')
    return macro_f1, micro_f1
    # TODO: experiment with different scores
    all_f1_scores = []
    for pred, target in zip(predictions, targets):
        f1 = hierarchical_f1_score(pred, target)
        all_f1_scores.append(f1)

    macro_f1 = np.mean(all_f1_scores)
    return macro_f1

def plot_training_metrics(train_history):
    plt.figure(figsize=(10, 12))

    # Plot 1: Loss
    plt.subplot(3, 1, 1)
    plt.plot(train_history['train_losses'], label='Train Loss', color='blue')
    plt.plot(train_history['val_losses'], label='Validation Loss', color='red')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Plot 2: Macro F1
    plt.subplot(3, 1, 2)
    plt.plot(train_history['train_macro_f1_scores'], label='Train Macro F1', color='blue')
    plt.plot(train_history['val_macro_f1_scores'], label='Validation Macro F1', color='red')
    plt.title('Macro F1 Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Macro F1 Score')
    plt.legend()
    plt.grid()

    # Plot 3: Micro F1
    plt.subplot(3, 1, 3)
    plt.plot(train_history['train_micro_f1_scores'], label='Train Micro F1', color='blue')
    plt.plot(train_history['val_micro_f1_scores'], label='Validation Micro F1', color='red')
    plt.title('Micro F1 Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Micro F1 Score')
    plt.legend()
    plt.grid()

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



class TextAugmenter:
    def __init__(self, prob=0.3):
        self.prob = prob

    def synonym_replacement(self, words):
        new_words = words.copy()
        for i, word in enumerate(words):
            if random.random() < self.prob:
                synonyms = wordnet.synsets(word)
                if synonyms:
                    lemmas = synonyms[0].lemmas()
                    if lemmas:
                        synonym = lemmas[0].name().replace("_", " ")
                        new_words[i] = synonym
        return new_words

    def random_deletion(self, words):
        if len(words) == 1:
            return words
        return [w for w in words if random.random() > self.prob]

    def random_swap(self, words):
        new_words = words.copy()
        if len(words) < 2:
            return new_words
        for _ in range(1):
            idx1, idx2 = random.sample(range(len(words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return new_words

    def random_case(self, words):
        return [w.upper() if random.random() < 0.1 else w for w in words]

    def __call__(self, text):
        words = text.strip().split()
        if random.random() < self.prob:
            words = self.synonym_replacement(words)
        if random.random() < self.prob:
            words = self.random_deletion(words)
        if random.random() < self.prob:
            words = self.random_swap(words)
        if random.random() < self.prob:
            words = self.random_case(words)
        return " ".join(words)