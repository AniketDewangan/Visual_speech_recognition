# data_preprocessing_dlib.py

import os
import cv2
import dlib
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

# ===============================
#     Helper Functions
# ===============================

def read_align_file(align_file_path):
    """
    Reads the .align file and extracts the word and pause segments.
    Returns a list of tuples: (start_frame, end_frame, label)
    Label is 1 for word, 0 for pause.
    """
    segments = []
    with open(align_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                start_time, end_time, token = parts
                # Convert time units to frame numbers (since they are multiples of 1000)
                start_frame = int(start_time) // 1000
                end_frame = int(end_time) // 1000
                label = 1 if token not in ['sil', 'sp'] else 0
                segments.append((start_frame, end_frame-1, label))
    return segments

# Initialize Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor_path = '/home/meghali/Documents/01/video_frame/Aniket/shape_predictor_68_face_landmarks.dat'  # Update this path if necessary
predictor = dlib.shape_predictor(predictor_path)  # Download this file separately

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
#         Data Preprocessing
# ===============================

def preprocess_and_save_data(root_dir, save_dir):
    """
    Preprocesses the videos and saves the extracted lip frames and labels.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    video_dir = os.path.join(root_dir, 'video')
    align_dir = os.path.join(root_dir, 'align')
    speakers = os.listdir(video_dir)
    data_entries = []
    for speaker in speakers:
        speaker_video_dir = os.path.join(video_dir, speaker)
        speaker_align_dir = os.path.join(align_dir, speaker)
        if not os.path.isdir(speaker_video_dir):
            continue
        video_files = [f for f in os.listdir(speaker_video_dir) if f.endswith('.mpg')]
        for video_file in video_files:
            video_path = os.path.join(speaker_video_dir, video_file)
            align_file = video_file.replace('.mpg', '.align')
            align_path = os.path.join(speaker_align_dir, align_file)
            if os.path.exists(align_path):
                # Extract frames
                frames, fps = extract_lip_frames(video_path)
                num_frames = frames.shape[0]

                # Read alignment segments
                segments = read_align_file(align_path)

                # Generate per-frame labels
                frame_labels = np.zeros(num_frames, dtype=np.int64)
                for start_frame, end_frame, label in segments:
                    if label == 1:
                        frame_labels[start_frame:end_frame] = 1

                # Save the data
                data_entry = {
                    'frames': frames,
                    'labels': frame_labels,
                    'fps': fps,
                    'video_file': video_file,
                    'speaker': speaker
                }
                data_entries.append(data_entry)
                print(f"Processed {video_file}")
            else:
                print(f'Alignment file not found for {video_file}')

    # Save all data entries to a file
    with open(os.path.join(save_dir, 'preprocessed_data.pkl'), 'wb') as f:
        pickle.dump(data_entries, f)
    print("Data preprocessing completed.")

# Example usage
if __name__ == "__main__":
    # Set the root directory of the GRID dataset
    root_dir = '/home/meghali/Documents/01/video_frame/Aniket'  # Change this to your GRID dataset path
    # Set the directory where preprocessed data will be saved
    save_dir = '/home/meghali/Documents/01/video_frame/Aniket/Frame1'
    preprocess_and_save_data(root_dir, save_dir)
