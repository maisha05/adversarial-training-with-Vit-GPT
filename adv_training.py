import os
import datasets
from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'

import nltk
import torch

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor

image_encoder_model = "google/vit-base-patch16-224-in21k"
text_decode_model = "gpt2"

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    image_encoder_model, text_decode_model)

model = model.to(device)

# Freezing the encoder
# for param in model.encoder.parameters():
#     param.requires_grad = False

# # Freeze GPT-2 decoder
# for param in model.decoder.parameters():
#     param.requires_grad = False

# image feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
# text tokenizer
tokenizer = AutoTokenizer.from_pretrained(text_decode_model)

# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
tokenizer.pad_token = tokenizer.eos_token

# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

output_dir = "/data/rashidm/vit-gpt/vit-gpt-model"
model.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image


class Flickr30kDataset(Dataset):
    def __init__(self, csv_file, img_dir, delimiter='",', transform=None, limit=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            delimiter (string): Delimiter used in the captions column to separate captions.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.data = []

        # Load and process the CSV file
        df = pd.read_csv(csv_file)
        if limit is not None:
            df = df.head(limit)  # Limit the number of rows processed

        for _, row in df.iterrows():
            filename = row['filename']
            image_path = os.path.join(self.img_dir, filename)
            captions = row['raw'].split(delimiter)
            img_id = row['img_id']
            for caption in captions:
                self.data.append({
                    'filename': filename,
                    'image_path': image_path,
                    'caption': caption,
                    'img_id': img_id
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Update item with transformed image
        item['image'] = image
        return item


# Example usage:
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

csv_file = '/home/rashidm/test code/vit-bert/flickr_annotations_30k.csv'
img_dir = '/data/rashidm/flickr30k/flickr30k-images'

flickr30k_dataset = Flickr30kDataset(csv_file=csv_file, img_dir=img_dir, transform=transform, limit=5000)

from datasets import Dataset
import pandas as pd
import torch

# Assuming flickr30k_dataset is your PyTorch dataset instance

# Step 1: Extract data from your PyTorch dataset
data = {
    'image_path': [],
    'caption': [],
    'filename': [],
    'img_id': []  # If you have image IDs
}

for item in flickr30k_dataset:
    data['image_path'].append(item['image_path'])
    data['caption'].append(item['caption'])
    data['filename'].append(item['filename'])
    data['img_id'].append(item['img_id'])  # If you have image IDs

# Step 2: Convert to Hugging Face dataset
hf_dataset = Dataset.from_dict(data)

# If you want to split the dataset into train/test splits, you can do so as follows:
train_test_split = hf_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

train_validation_split = train_dataset.train_test_split(test_size=0.1)  # Adjust the test_size as needed
validation_dataset = train_validation_split['test']
train_dataset = train_validation_split['train']


# Now you have a Hugging Face dataset


# text preprocessing step
def tokenization_fn(captions, max_target_length):
    """Run tokenization on captions."""

    # Make sure captions is a list, even if there's only one caption
    if not isinstance(captions, list):
        captions = [captions]

    labels = tokenizer(captions,
                       padding="max_length",
                       max_length=max_target_length,
                       truncation=True).input_ids

    return labels


def feature_extraction_fn(image_paths, check_image=True):
    images = []
    if check_image:
        for image_file in image_paths:
            try:
                # Open the image file and append to the list
                img = Image.open(image_file).convert("RGB")  # Ensure the image is RGB
                images.append(img)
            except Exception as e:
                # print(f"Failed to load image {image_file}: {e}")
                continue  # Skip images that fail to load
    else:
        for image_file in image_paths:
            images.append(Image.open(image_file).convert("RGB"))

    # Process images through the feature extractor
    encoder_inputs = feature_extractor(images=images, return_tensors="pt")

    return encoder_inputs["pixel_values"]


# Adversarial example generation function
def generate_adversarial_example(model, pixel_values, labels, epsilon=0.01):
    # Ensure model is in evaluation mode
    model.eval()
    labels[
        labels == tokenizer.pad_token_id] = -100  # Correctly setting pad token labels for ignoring in loss calculation

    # Enable gradient calculation
    pixel_values.requires_grad = True

    # Forward pass and calculate loss
    outputs = model(pixel_values=pixel_values, labels=labels)
    loss = outputs.loss

    # print('ATTACK!!!!!!')
    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()

    # Apply FGSM
    pixel_values_grad = pixel_values.grad.data
    adversarial_pixel_values = pixel_values + epsilon * pixel_values_grad.sign()
    adversarial_pixel_values = torch.clamp(adversarial_pixel_values, 0, 1)
    adversarial_pixel_values = adversarial_pixel_values.to(device)

    return adversarial_pixel_values.detach()  # Detach to stop tracking gradients


def preprocess_fn(examples, max_target_length, check_image=True, generate_adversarial=False, epsilon=0.01):
    # """Run tokenization + image feature extraction"""
    image_paths = examples['image_path']
    captions = examples['caption']

    model_inputs = {}
    labels = tokenizer(captions, padding="max_length", max_length=max_target_length, truncation=True,
                       return_tensors="pt").input_ids
    model_inputs['labels'] = labels

    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    encoder_inputs = feature_extractor(images=images, return_tensors="pt")
    pixel_values = encoder_inputs["pixel_values"]
    pixel_values = pixel_values.to(device)
    labels = labels.to(device)

    if generate_adversarial:
        # Generate adversarial examples for each image individually
        for i in range(pixel_values.size(0)):
            curr_pixel_values = pixel_values[i].unsqueeze(0)
            curr_labels = labels[i].unsqueeze(0)
            adv_pixel_values = generate_adversarial_example(model, curr_pixel_values, curr_labels, epsilon=epsilon)
            if i == 0:
                model_inputs['pixel_values'] = adv_pixel_values
            else:
                model_inputs['pixel_values'] = torch.cat((model_inputs['pixel_values'], adv_pixel_values), dim=0)
    else:
        model_inputs['pixel_values'] = pixel_values

    return model_inputs


# processed_dataset = hf_dataset.map(
#     function=preprocess_fn,
#     batched=True,
#     fn_kwargs={"max_target_length": 128},
#     #remove_columns=ds['train'].column_names
# )

# Now, apply the preprocess_fn to each of these splits
def apply_preprocessing(dataset, **kwargs):
    return dataset.map(
        function=preprocess_fn,
        batched=True,
        fn_kwargs={"max_target_length": 128, **kwargs},
    )


processed_train_dataset = apply_preprocessing(train_dataset, generate_adversarial=True, epsilon=0.01)
processed_test_dataset = apply_preprocessing(test_dataset, generate_adversarial=False, epsilon=0.01)
processed_validation_dataset = apply_preprocessing(validation_dataset)

# print(processed_train_dataset)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    output_dir="/data/rashidm/vit-gpt/image-captioning-output-advtraining",
)

import evaluate

metric = evaluate.load("rouge")

import numpy as np
from nltk.translate.bleu_score import corpus_bleu

ignore_pad_token_for_loss = True


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)

    # Flatten labels for BLEU
    # Correcting the input format for corpus_bleu
    decoded_labels_flat = [[label.split()] for label in
                           decoded_labels]  # Each reference set contains one reference translation
    decoded_preds_split = [pred.split() for pred in decoded_preds]  # Each hypothesis is tokenized

    # Now calculate BLEU score correctly
    bleu_score = corpus_bleu(decoded_labels_flat, decoded_preds_split)

    # bleu_result = metric.compute(predictions=[pred.split() for pred in decoded_preds], references=[decoded_labels_flat])
    # bleu_result = {f"bleu": bleu_result["bleu"] * 100}

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels,
                            use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result["bleu"] = bleu_score
    return result


from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_validation_dataset,
    data_collator=default_data_collator,
)
trainer.train()

trainer.save_model("/data/rashidm/vit-gpt/image-captioning-output-advtrain")
tokenizer.save_pretrained("/data/rashidm/vit-gpt/image-captioning-output-advtrain")
print('done')
# Assuming compute_metrics function remains unchanged as it's already set up for BLEU calculation

# After the training process, evaluate the model on both training and testing datasets
train_results = trainer.evaluate(processed_train_dataset)
test_results = trainer.evaluate(processed_test_dataset)

# Optionally, save the results to a file or print them
print("Training Results:", train_results)
print("Testing Results:", test_results)

# Saving the results for further use
import json

with open("/data/rashidm/vit-gpt/image-captioning-output/train_results.json", "w") as f:
    json.dump(train_results, f)
with open("/data/rashidm/vit-gpt/image-captioning-output/test_results.json", "w") as f:
    json.dump(test_results, f)

print('Evaluation done and results saved.')

# from transformers import pipeline
# image_captioner = pipeline("image-to-text", model="./image-captioning-output")
# image_captioner("sample_image.png")
