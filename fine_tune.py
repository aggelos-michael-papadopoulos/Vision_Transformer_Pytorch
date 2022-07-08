from datasets import load_dataset
from transformers import ViTFeatureExtractor
from torchvision.transforms import (CenterCrop,
                                    Normalize,
                                    Compose,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)
from torch.utils.data import DataLoader
import torch
from transformers import ViTForImageClassification
from transformers import TrainingArguments, Trainer
from datasets import load_metric
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# load cifar10 (only small portion for demonstration purposes)
train_ds, test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:2000]'])

# split up training into training and validation
splits = train_ds.train_test_split(test_size=0.1)
# split up training into training + validation
train_ds = splits['train']
val_ds = splits['test']

# Know our classes
id2label = {id: label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label: id for id, label in id2label.items()}

# Data preprocessing
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
_train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

# Set the transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)

# load preprocessed images on the fly
# train_ds[0]

# create the Dataloader
def collate_fn(examples):
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {'pixel_values': pixel_values, 'labels': labels}


train_loader = DataLoader(train_ds, collate_fn=collate_fn ,batch_size=4)

batch = next(iter(train_loader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)



# But we are not using dataloaders --> HuggingFace train API

# Define the model

# Here we define the model.
# We define a ViTForImageClassification, which places a linear layer (nn.Linear) on top of a pre-trained ViTModel.
# The linear layer is placed on top of the last hidden state of the [CLS] token,
# which serves as a good representation of an entire image.
# The model itself is pre-trained on ImageNet-21k, a dataset of 14 million labeled images.
# We also specify the number of output neurons (which is 10, in case of CIFAR-10),
# and we set the id2label and label2id mapping, which we be added as attributes to the configuration of the model
# (which can be accessed as model.config).

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  num_labels=10,
                                                  id2label=id2label,
                                                  label2id=label2id)
# print(model)

# Here we set the evaluation to be done at the end of each epoch, tweak the learning rate,
# set the training and evaluation batch_sizes and customize the number of epochs for training,
# as well as the weight decay.
metric_name = "accuracy"

args = TrainingArguments(
    f"test-cifar-10",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
)

# We also define a compute_metrics function that will be used to compute metrics at evaluation.
# We use "accuracy", which is available in HuggingFace Datasets.

metric = load_metric('accuracy')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Then we just need to pass all of this along with our datasets to the Trainer:
trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
)

# train the model
trainer.train()

# evaluation
outputs = trainer.predict(test_ds)

print(outputs.metrics['test_accuracy'])

# Confusion matrix:
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

labels = train_ds.features['label'].names
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)