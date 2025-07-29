from datasets import load_dataset
from datasets import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from MobileNetv1 import *
import torch.nn as nn
import torch.optim as optim



dataset = load_dataset("zh-plus/tiny-imagenet")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize from 64x64 to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


subset_train_size = len(dataset["train"]) // 1000 # 100 images
subset_val_size = len(dataset["valid"]) // 100   # 100 images

# Select a subset of the dataset
train_indices = list(range(subset_train_size))
val_indices = list(range(subset_val_size))



def transform_example(examples):
    # Check if 'examples["image"]' is a list (meaning it's a batch of images)
    if isinstance(examples["image"], list):
        # If it's a list, iterate through each image in the list
        examples["image"] = [transform(img.convert("RGB")) for img in examples["image"]]
    else:
        # If it's a single image (e.g., PIL Image object), process it directly
        examples["image"] = transform(examples["image"].convert("RGB"))
    return examples


dataset.set_transform(transform_example)


#get train and val datasets
train_dataset = dataset["train"].select(train_indices)
val_dataset = dataset["valid"].select(val_indices)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


device = torch.device('cpu')


model = MobileNetV1(num_classes=200).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0

    for batch in dataloader:
        images, labels = batch["image"], batch["label"]

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels)

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct.double() / len(dataloader.dataset)
    return epoch_loss, accuracy.item()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch["image"], batch["label"]

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct.double() / len(dataloader.dataset)
    return epoch_loss, accuracy.item()

num_epochs = 10
print("fetched and preprocessed data, beginnig training")
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "./model")


