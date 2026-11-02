# Install gdown automatically if it is not available.
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("gdown") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

# Whale Detection v3: code cells extracted from the Colab notebook.
# Notebook cell order is preserved. Set WHALE_DATA_DIR to your local data folder.

# --- Notebook code cell 2 ---
# Imports - feel free to add other useful libraries!

import os
import subprocess
import zipfile
import io
import math
import torch
import torchaudio
import librosa
import librosa.display
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Audio, display
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import nn
from torch.utils.data import Dataset, DataLoader


# --- Notebook code cell 4 ---
subprocess.run([
    sys.executable, "-m", "gdown", "--fuzzy",
    "https://drive.google.com/file/d/1CoayH6jchCxKLCPzdKLzCM5lKUETGClq/view?usp=drive_link",
    "--output", "whale_data.zip"
], check=True)
with zipfile.ZipFile("whale_data.zip") as archive:
    archive.extractall(".")


# --- Notebook code cell 6 ---
data_root = os.environ.get("WHALE_DATA_DIR", "whale-detection-challenge/data")
audio_folder = os.path.join(data_root, "audio")
labels_path = os.path.join(data_root, "labels.csv")

# Load labels
labels_df = pd.read_csv(labels_path)

# Show first 10 rows
labels_df[:10]


# --- Notebook code cell 8 ---
class WhaleDataset(Dataset):
    def __init__(self, audio_dir, labels_df):
        self.audio_dir = audio_dir
        self.labels_df = labels_df

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.labels_df.iloc[idx, 0])
        label = self.labels_df.iloc[idx, 1]
        audio, sr = torchaudio.load(audio_path)
        return audio.flatten(), torch.tensor(label, dtype=torch.float32)

# Create dataset
full_dataset = WhaleDataset(audio_folder, labels_df)

# Split the data
train_data, test_data = train_test_split(full_dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Create data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print(f"Train set: {len(train_data)} files")
print(f"Validation set: {len(val_data)} files")
print(f"Test set: {len(test_data)} files")


# --- Notebook code cell 9 ---
print("Training example looks like: ", train_data[0])
print("Length of input: ", train_data[0][0].shape)


# --- Notebook code cell 11 ---
def display_audio_players(labels_df, audio_folder, num_examples=12, label=1):
    # Filter file names based on the label
    file_names = labels_df[labels_df.iloc[:, 1] == label].iloc[:, 0].tolist()

    # Loop through first num_examples file names
    for i, file_name in enumerate(file_names[:num_examples]):
        file_path = os.path.join(audio_folder, file_name)

        # Load the audio file
        with open(file_path, 'rb') as fd:
            contents = fd.read()
        audio, sr = librosa.load(io.BytesIO(contents))

        print(f"{'Whale' if label == 1 else 'Non-Whale'} Example {i+1}: {file_name}")
        display(Audio(data=audio, rate=sr, autoplay=False))


# --- Notebook code cell 12 ---
# TODO: play around with the number of examples and change the label from 0 (non-whale)
# to 1 (whale) in order to listen to whale calls

display_audio_players(labels_df, audio_folder, num_examples=12, label=0)


# --- Notebook code cell 14 ---
def display_waveform_examples(labels_df, audio_folder, num_examples=12, label=1):
    file_names = labels_df[labels_df.iloc[:, 1] == label].iloc[:, 0].tolist()

    fig, axes = plt.subplots(math.ceil(num_examples/3), 3, figsize=(3*5, 5*num_examples//3))
    for i, file_name in enumerate(file_names[:num_examples]):
        file_path = os.path.join(audio_folder, file_name)
        with open(file_path, 'rb') as fd:
            contents = fd.read()
        audio, sr = librosa.load(io.BytesIO(contents))
        ax = axes[i//3, i%3]
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        ax.set_title(f"{'Whale' if label == 1 else 'Non-Whale'} Example {i+1}: {file_name}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


# --- Notebook code cell 15 ---
# TODO: play around with the number of examples and the label (0 or 1)

display_waveform_examples(labels_df, audio_folder, num_examples=12, label=0)


# --- Notebook code cell 17 ---
def display_spectrogram_examples(labels_df, audio_folder, num_examples=9, label=1):
    file_names = labels_df[labels_df.iloc[:, 1] == label].iloc[:, 0].tolist()

    fig, axes = plt.subplots(math.ceil(num_examples/3), 3, figsize=(3*5, 4*num_examples//3))
    for i, file_name in enumerate(file_names[:num_examples]):
        file_path = os.path.join(audio_folder, file_name)
        with open(file_path, 'rb') as fd:
            contents = fd.read()
        audio, sr = librosa.load(io.BytesIO(contents))

        # Compute and display spectrogram
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        ax = axes[i//3, i%3]
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax)
        ax.set_ylim(0, 1000)
        ax.set_title(f"{'Whale' if label == 1 else 'Non-Whale'} Spectrogram {i+1}")
        fig.colorbar(img, ax=ax, format='%+2.0f dB')

    plt.tight_layout()
    plt.show()


# --- Notebook code cell 18 ---
# TODO: play around with the number of examples and the label (0 or 1)

display_spectrogram_examples(labels_df, audio_folder, num_examples=12, label=0)


# --- Notebook code cell 21 ---
class WhaleNN(nn.Module):
    def __init__(self):
        super(WhaleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4000, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

model = WhaleNN()


# --- Notebook code cell 23 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # TODO: Try changing the learning rate

num_epochs = 20 # TODO: Try changing the number of epochs
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Calculate average training loss for this epoch
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            val_loss += criterion(outputs, batch_labels.unsqueeze(1)).item()

    # Calculate average validation loss for this epoch
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


# --- Notebook code cell 24 ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()


# --- Notebook code cell 26 ---
def evaluate_model(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            predictions = (outputs > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)

# Get predictions and true labels
predictions, true_labels = evaluate_model(model, test_loader, device)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f} \n")

# Create confusion matrix
cm = confusion_matrix(true_labels, predictions)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# --- Notebook code cell 28 ---
def find_misclassified(model, data_loader, device):
    model.eval()
    misclassified = []
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            predictions = (outputs > 0.5).float()

            # Find indices of misclassified examples
            misclassified_mask = predictions.squeeze() != batch_labels
            misclassified_indices = misclassified_mask.nonzero(as_tuple=True)[0]

            for idx in misclassified_indices:
                misclassified.append({
                    'features': batch_features[idx].cpu().numpy(),
                    'true_label': batch_labels[idx].item(),
                    'predicted': predictions[idx].item(),
                    'confidence': outputs[idx].item()
                })

    return misclassified

# Find misclassified examples
misclassified = find_misclassified(model, test_loader, device)

def display_misclassified(misclassified, num_examples=10):
    np.random.shuffle(misclassified)
    for i, example in enumerate(misclassified[:num_examples]):
        audio = example['features']
        true_label = "Whale" if example['true_label'] == 1 else "Non-Whale"
        predicted = "Whale" if example['predicted'] == 1 else "Non-Whale"
        confidence = example['confidence']

        # Plot waveform
        librosa.display.waveshow(audio, sr=2000)
        plt.title(f"Waveform (True: {true_label}, Predicted: {predicted}, Confidence: {confidence:.2f})")

        plt.tight_layout()
        plt.show()

        # Play the audio
        wav_object = io.BytesIO(bytes())
        scipy.io.wavfile.write(wav_object, 2000, audio)
        wav_object.seek(0)
        audio, sr = librosa.load(wav_object)
        display(Audio(data=audio, rate=sr, autoplay=False))

        print(f"Example {i+1}:")
        print(f"True Label: {true_label}")
        print(f"Predicted: {predicted}")
        print(f"Model Confidence: {confidence:.2f}")
        print("-----------------------------\n")

# Display misclassified examples
display_misclassified(misclassified, num_examples = 10)


# --- Notebook code cell 31 ---
def create_spectrogram(audio, sr=2000, n_mels=128, fmax=1000, hop_length=512):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram


# --- Notebook code cell 33 ---
spectrograms = []
filenames = []
for _, row in labels_df.iterrows():
    audio_path = os.path.join(audio_folder, row['clip_name'])
    filenames.append(row['clip_name'])
    audio, sr = torchaudio.load(audio_path)
    audio = audio.numpy()[0]
    spectrogram = create_spectrogram(audio, sr, n_mels=128, fmax=1000, hop_length=128)
    spectrograms.append(spectrogram)

all_spectrograms = np.array(spectrograms)
all_filenames = np.array(filenames)


# --- Notebook code cell 35 ---
all_labels = labels_df['label'].values


# --- Notebook code cell 37 ---
import random

def time_shift(spectrogram, max_shift=20):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(spectrogram, shift, axis=1)

def frequency_shift(spectrogram, max_shift=10):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(spectrogram, shift, axis=0)

def add_noise(spectrogram, noise_factor=0.005):
    noise = np.random.randn(*spectrogram.shape)
    return spectrogram + noise_factor * noise

def augment_spectrogram(spectrogram):
    if random.random() < 0.5:
        spectrogram = time_shift(spectrogram)
    if random.random() < 0.5:
        spectrogram = frequency_shift(spectrogram)
    if random.random() < 0.5:
        spectrogram = add_noise(spectrogram)
    return spectrogram

def create_augmented_data(spectrograms, labels, filenames, num_augmentations=2):
    augmented_spectrograms = []
    augmented_labels = []
    augmented_filenames = []
    for spectrogram, label, filename in zip(spectrograms, labels, filenames):
        for _ in range(num_augmentations):
            augmented_spectrograms.append(augment_spectrogram(spectrogram))
            augmented_labels.append(label)
            augmented_filenames.append(filename)
    return np.array(augmented_spectrograms), np.array(augmented_labels), np.array(augmented_filenames)


# --- Notebook code cell 38 ---
class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels, filenames, augment=False):
        self.spectrograms = spectrograms
        self.labels = labels
        self.augment = augment
        self.filenames = filenames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spectrogram = self.spectrograms[idx]
        label = self.labels[idx]
        filename = self.filenames[idx]

        if self.augment:
            spectrogram = augment_spectrogram(spectrogram)

        spectrogram = np.expand_dims(spectrogram, axis=0)
        return torch.from_numpy(spectrogram).float(), torch.tensor(label, dtype=torch.float32), filename

# Split the data
train_spectrograms, temp_spectrograms, train_labels, temp_labels, train_filenames, temp_filenames = train_test_split(
    all_spectrograms, all_labels, all_filenames, test_size=0.4, random_state=42, stratify=all_labels
)
val_spectrograms, test_spectrograms, val_labels, test_labels, val_filenames, test_filenames = train_test_split(
    temp_spectrograms, temp_labels, temp_filenames, test_size=0.5, random_state=42, stratify=temp_labels
)


# --- Notebook code cell 39 ---
# Create augmented training data
augmented_spectrograms, augmented_labels, augmented_filenames = create_augmented_data(train_spectrograms, train_labels, train_filenames, num_augmentations=2)

# Append augmented data to original training data
train_spectrograms_augmented = np.concatenate([train_spectrograms, augmented_spectrograms], axis=0)
train_labels_augmented = np.concatenate([train_labels, augmented_labels], axis=0)
train_filenames_augmented = np.concatenate([train_filenames, augmented_filenames], axis=0)

# Create datasets
train_data_augmented = SpectrogramDataset(train_spectrograms_augmented, train_labels_augmented, train_filenames_augmented, augment=True)
val_data = SpectrogramDataset(val_spectrograms, val_labels, val_filenames, augment=False)
test_data = SpectrogramDataset(test_spectrograms, test_labels, test_filenames, augment=False)

# Create data loaders
train_loader = DataLoader(train_data_augmented, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print(f"Train set: {len(train_data_augmented)} files")
print(f"Validation set: {len(val_data)} files")
print(f"Test set: {len(test_data)} files")


# --- Notebook code cell 41 ---
print(f'Spectrogram shape: {train_data_augmented[0][0].shape}')
print(f'Lable shape: {train_data_augmented[0][1].shape}')
print(f'First training example: {train_data_augmented[0]}')


# --- Notebook code cell 43 ---
class CNN(nn.Module):
    def __init__(self, input_shape):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the size of the flattened features
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            self.flat_features = self.num_flat_features(self.features(sample_input))

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten while keeping batch size
        x = self.classifier(x)
        return x

model = CNN(train_data_augmented[0][0].shape)
print(f"Number of flat features: {model.flat_features}")


# --- Notebook code cell 45 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # TODO: Experiment with learning rate (lr)

num_epochs = 10   # TODO: Experiment with number of epochs
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for spectrograms, labels, filenames in train_loader:
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(spectrograms)
        loss = criterion(outputs.reshape(-1), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for spectrograms, labels, filenames in val_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            val_loss += criterion(outputs, labels.unsqueeze(1)).item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')


# --- Notebook code cell 47 ---
def evaluate_model(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for spectrograms, labels, filenames in data_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            predictions = (outputs > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend(filenames)

    return np.array(all_predictions), np.array(all_labels)

# Get predictions and true labels
predictions, true_labels = evaluate_model(model, test_loader, device)
# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"Testing Accuracy: {accuracy:.4f} \n")

# Create confusion matrix
cm = confusion_matrix(true_labels, predictions)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# --- Notebook code cell 49 ---
def find_misclassified(model, data_loader, device, confidence_threshold=0.9):
    model.eval()
    misclassified = []
    with torch.no_grad():
        for spectrograms, labels, filenames in data_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            predictions = (outputs > 0.5).float()
            confidences = torch.max(outputs, 1 - outputs)  # Confidence in the prediction
            threshold = confidence_threshold * torch.ones_like(confidences)

            # Find indices of misclassified examples
            misclassified_mask = (predictions.squeeze() != labels) & (confidences.squeeze() > threshold.squeeze())
            misclassified_indices = misclassified_mask.nonzero(as_tuple=True)[0].tolist()

            for idx in misclassified_indices:
                misclassified.append({
                    'features': spectrograms[idx].cpu().numpy(),
                    'true_label': labels[idx].item(),
                    'predicted': predictions[idx].item(),
                    'confidence': confidences[idx].item(),
                    'filename': filenames[idx]
                })

    misclassified.sort(key=lambda x: x['confidence'], reverse=True)
    return misclassified

def display_misclassified(misclassified, num_examples=10):
    for i, example in enumerate(misclassified[:num_examples]):
        spectrogram = example['features']
        true_label = "Whale" if example['true_label'] == 1 else "Non-Whale"
        predicted = "Whale" if example['predicted'] == 1 else "Non-Whale"
        confidence = example['confidence']
        filename = example['filename']

        plt.figure(figsize=(5,4))
        plt.imshow(spectrogram.squeeze(), aspect='auto', cmap='plasma', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"True: {'Whale' if example['true_label'] == 1 else 'Non-Whale'}, " +
                  f"Predicted: {'Whale' if example['predicted'] == 1 else 'Non-Whale'}, " +
                  f"Confidence: {example['confidence']:.2f}, "+ "File:"+filename)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()
        file_path = os.path.join(audio_folder, filename)

        # Load the audio file
        with open(file_path, 'rb') as fd:
            contents = fd.read()
        audio, sr = librosa.load(io.BytesIO(contents))

        print(f"True: {'Whale' if example['true_label'] == 1 else 'Non-Whale'}, " +
                  f"Predicted: {'Whale' if example['predicted'] == 1 else 'Non-Whale'}, " +
                  f"Confidence: {example['confidence']:.2f}, "+ "File:"+filename)
        display(Audio(data=audio, rate=sr, autoplay=False))

def correct_labels(original_labels_df, misclassified_examples):
    # Create a copy of the original DataFrame
    corrected_labels_df = original_labels_df.copy()

    for example in misclassified_examples:
        filename = example['filename']
        predicted_label = example['predicted']

        # Find the index of the filename in the DataFrame
        idx = corrected_labels_df.index[corrected_labels_df['clip_name'] == filename].tolist()

        if idx:
            # Update the label
            corrected_labels_df.loc[idx[0], 'label'] = predicted_label

    return corrected_labels_df


# --- Notebook code cell 50 ---
# TODO: Toggle between train_loader, val_loader, and test_loader
misclassified = find_misclassified(model, train_loader, device)
print("Number of confidently misclassified examples: ", len(misclassified))
display_misclassified(misclassified)


# --- Notebook code cell 53 ---
misclassified = find_misclassified(model, train_loader, device)
user_input = input("Do you want to correct these labels? (yes/no): ").lower()

if user_input == 'yes':
    # Correct labels
    corrected_labels_df = correct_labels(labels_df, misclassified)

    # Print summary of changes
    num_changes = (corrected_labels_df['label'] != labels_df['label']).sum()
    print(f"Number of labels corrected: {num_changes}")

    labels_df = corrected_labels_df

else:
    print("Labels were not changed.")


# --- Notebook code cell 56 ---
import torchvision.models as models

class WhaleCallResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(WhaleCallResNet, self).__init__()
        self.resnet = models.resnet18(weights='DEFAULT')
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)

model = WhaleCallResNet()


# --- Notebook code cell 58 ---
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

# Create an ensemble of different models
model1 = CNN(train_data_augmented[0][0].shape)
model2 = CNN(train_data_augmented[0][0].shape)
model3 = WhaleCallResNet()
model = EnsembleModel([model1, model2, model3])
