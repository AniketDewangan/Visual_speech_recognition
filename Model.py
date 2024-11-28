# model_training_and_testing.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.cuda.amp import GradScaler, autocast
import pickle

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===============================
#        Dataset Class
# ===============================

class LipReadingDataset(Dataset):
    def __init__(self, data_entries):
        self.data_entries = data_entries

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        data_entry = self.data_entries[idx]
        frames = data_entry['frames']  # NumPy array of shape (num_frames, 50, 100)
        frame_labels = data_entry['labels']  # NumPy array of shape (num_frames,)
        # Normalize frames
        frames = frames / 255.0
        frames = torch.FloatTensor(frames)
        frames = frames.unsqueeze(1)  # Add channel dimension
        frame_labels = torch.LongTensor(frame_labels)
        return frames, frame_labels

# ===============================
#       Collate Function
# ===============================

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    """
    videos, frame_labels = zip(*batch)
    batch_size = len(videos)

    # Get sequence lengths
    input_lengths = [v.size(0) for v in videos]

    # Pad videos and frame labels
    max_video_len = max(input_lengths)
    padded_videos = torch.zeros(batch_size, max_video_len, 1, 50, 100)
    padded_frame_labels = torch.zeros(batch_size, max_video_len, dtype=torch.long)
    for i, (v, fl) in enumerate(zip(videos, frame_labels)):
        length = v.size(0)
        padded_videos[i, :length, :, :, :] = v
        padded_frame_labels[i, :length] = fl

    return padded_videos, padded_frame_labels, input_lengths

# ===============================
#        Model Definition
# ===============================

class WordBoundaryNet(nn.Module):
    def __init__(self):
        super(WordBoundaryNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(96)
        self.pool3 = nn.MaxPool3d((1, 2, 2))

        # Recurrent layers
        self.gru1 = nn.GRU(96 * 6 * 12, 256, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(512, 256, bidirectional=True, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(512, 2)  # 2 classes: blank (0) and word (1)

    def forward(self, x, lengths):
        # x shape: (batch_size, time_steps, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)  # (batch_size, channels, time_steps, height, width)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Prepare for RNN
        batch_size, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # (batch_size, time_steps, channels, height, width)
        x = x.contiguous().view(batch_size, t, -1)  # Flatten

        # Pack the sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Recurrent layers
        packed_output, _ = self.gru1(packed_input)
        packed_output, _ = self.gru2(packed_output)

        # Unpack sequences
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        x = self.fc(x)  # Shape: (batch_size, time_steps, num_classes)
        # Do not apply log_softmax here since CrossEntropyLoss expects raw logits

        return x  # Shape: (batch_size, time_steps, num_classes)

# ===============================
#        Training Function
# ===============================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    scaler = GradScaler()  # For mixed precision training

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        for batch_idx, (videos, frame_labels, input_lengths) in enumerate(train_loader):
            videos = videos.to(device)
            frame_labels = frame_labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(videos, input_lengths)  # Outputs shape: (batch_size, time_steps, num_classes)

                # Flatten outputs and labels
                outputs_flat = outputs.view(-1, 2)
                labels_flat = frame_labels.view(-1)

                # Compute loss
                loss = criterion(outputs_flat, labels_flat)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # For accuracy metrics
            preds_flat = outputs_flat.argmax(1)
            all_preds.extend(preds_flat.cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_loss:.4f}')
        print(f'Training Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0
        val_all_preds = []
        val_all_labels = []
        with torch.no_grad():
            for videos, frame_labels, input_lengths in val_loader:
                videos = videos.to(device)
                frame_labels = frame_labels.to(device)

                outputs = model(videos, input_lengths)  # Outputs shape: (batch_size, time_steps, num_classes)

                # Flatten outputs and labels
                outputs_flat = outputs.view(-1, 2)
                labels_flat = frame_labels.view(-1)

                # Compute loss
                loss = criterion(outputs_flat, labels_flat)
                val_loss += loss.item()

                # For accuracy metrics
                preds_flat = outputs_flat.argmax(1)
                val_all_preds.extend(preds_flat.cpu().numpy())
                val_all_labels.extend(labels_flat.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_all_labels, val_all_preds)
        val_precision, val_recall, val_f1_score, _ = precision_recall_fscore_support(val_all_labels, val_all_preds, average='binary')
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1_score:.4f}')

# ===============================
#           Inference
# ===============================

def test_on_video(model, data_entry):
    """
    Test the trained model on a specific preprocessed video data entry.
    """
    model.eval()
    with torch.no_grad():
        frames = data_entry['frames']
        fps = data_entry['fps']
        num_frames = frames.shape[0]
        frames = frames / 255.0
        frames = torch.FloatTensor(frames).unsqueeze(0).unsqueeze(2)  # Shape: (1, time_steps, 1, 50, 100)
        input_length = [num_frames]

        frames = frames.to(device)
        outputs = model(frames, input_length)
        outputs = outputs.squeeze(0)  # Shape: (time_steps, num_classes)
        output_probs = F.softmax(outputs, dim=1)
        pred_labels = output_probs.argmax(dim=1).cpu().numpy()

        # Convert frame indices to time
        frame_times = np.arange(num_frames) / fps

        # Find contiguous word segments
        word_segments = []
        in_word = False
        for i, label in enumerate(pred_labels):
            if label == 1 and not in_word:
                # Start of a word
                start_time = frame_times[i]
                in_word = True
            elif label == 0 and in_word:
                # End of a word
                end_time = frame_times[i]
                word_segments.append((start_time, end_time))
                in_word = False
        if in_word:
            # Handle case where video ends during a word
            end_time = frame_times[-1]
            word_segments.append((start_time, end_time))

        # Output the word segments
        for idx, (start, end) in enumerate(word_segments):
            print(f"word{idx+1} start time {start:.2f}s to end time {end:.2f}s")

# ===============================
#              Main
# ===============================

if __name__ == "__main__":
    # Load preprocessed data
    with open('/home/meghali/Documents/01/video_frame/Aniket/Frame/preprocessed_data.pkl', 'rb') as f:
        data_entries = pickle.load(f)

    # Create dataset
    dataset = LipReadingDataset(data_entries)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Compute class weights to handle imbalance
    total_frames = 0
    class_counts = np.zeros(2)
    for frames, frame_labels in train_dataset:
        total_frames += len(frame_labels)
        class_counts += np.bincount(frame_labels.numpy(), minlength=2)

    class_weights = total_frames / (2 * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Initialize model, criterion, optimizer
    model = WordBoundaryNet().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training parameters
    num_epochs = 15  # Adjust as needed

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # Save the model
    torch.save(model.state_dict(), '/home/meghali/Documents/01/video_frame/Aniket/Frame/word_boundary_model.pth')
    # model = torch.load('/home/meghali/Documents/01/video_frame/Aniket/Frame/word_boundary_model.pth', weights_only=False)
    # Example usage of testing
    # Select a data entry for testing (e.g., the first entry in the dataset)
    test_data_entry = data_entries[9]  # Change the index as needed
    test_on_video(model, test_data_entry)

    print("Testing completed.")
