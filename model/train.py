import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np

from cnn import EmotionCNN
from features import extract_mel_spectrogram

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful"]
LABEL_MAP = {e: i for i, e in enumerate(EMOTIONS)}
EPOCHS = 20
print("Starting training...")   

class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        for emotion in EMOTIONS:
            folder = os.path.join(data_dir, emotion)
            for file in os.listdir(folder):
                self.samples.append((os.path.join(folder, file), LABEL_MAP[emotion]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel = extract_mel_spectrogram(path)
        mel = torch.tensor(mel).unsqueeze(0)
        return mel, label


dataset = AudioDataset("data/ravdess")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")
model = EmotionCNN(num_classes=len(EMOTIONS)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 🔍 Validation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

    acc = 100 * correct / total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.2f} | "
        f"Val Accuracy: {acc:.2f}%"
    )

torch.save(model.state_dict(), "model/emotion_model.pt")
