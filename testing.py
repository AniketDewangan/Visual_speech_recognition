import os
import cv2
import dlib
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===============================
#     Helper Functions
# ===============================

# Initialize Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor_path = '/home/meghali/Documents/01/video_frame/Aniket/shape_predictor_68_face_landmarks.dat'  # Update this path if necessary
predictor = dlib.shape_predictor(predictor_path)  # Ensure this file is available

def extract_lip_frames(video_path):
    """
    Extracts the lip region frames from the video.
    Returns a numpy array of frames and the video's FPS.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = detector(gray)
        if len(faces) == 0:
            # If no face is detected, append a zero frame
            frames.append(np.zeros((50, 100), dtype=np.float32))
            continue
        # Assume the first face is the target
        face = faces[0]
        # Get facial landmarks
        landmarks = predictor(gray, face)
        # Get lip region coordinates
        lip_points = []
        for n in range(48, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            lip_points.append((x, y))
        lip_points = np.array(lip_points)
        x, y, w, h = cv2.boundingRect(lip_points)
        # Expand the bounding box a bit
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = w + 2 * margin
        h = h + 2 * margin
        lip_frame = gray[y:y+h, x:x+w]
        lip_frame = cv2.resize(lip_frame, (100, 50))
        frames.append(lip_frame.astype(np.float32))
    cap.release()
    if len(frames) == 0:
        # If no frames are detected, return zeros
        frames = np.zeros((1, 50, 100), dtype=np.float32)
    else:
        frames = np.stack(frames)  # Shape: (num_frames, height, width)
    return frames, fps

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
#           Inference
# ===============================

def test_on_video_file(model, video_path):
    """
    Process a video file, run the model, and output predicted word segments.
    """
    # Extract lip frames from the video
    frames, fps = extract_lip_frames(video_path)
    num_frames = frames.shape[0]

    # Preprocess the frames
    frames = frames / 255.0
    frames = torch.FloatTensor(frames).unsqueeze(0).unsqueeze(2)  # Shape: (1, time_steps, 1, 50, 100)
    input_length = [num_frames]

    # Send data to device
    frames = frames.to(device)

    # Run the model
    model.eval()
    with torch.no_grad():
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
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = WordBoundaryNet().to(device)

    # Load the model weights
    model_path = '/home/meghali/Documents/01/video_frame/Aniket/Frame/word_boundary_model.pth'  # Update the path
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Provide the video file path
    video_path = '/home/meghali/Documents/01/video_frame/Aniket/video/s31/bbac2n.mpg'  # Update the path

    # Run the test
    test_on_video_file(model, video_path)
