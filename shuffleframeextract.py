import cv2
import os
import random
from sklearn.model_selection import train_test_split

def extract_frames(video_path, output_folder, fps=30, target_frame_count=100):
    video_cap = cv2.VideoCapture(video_path)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    
    # Calculate frame interval
    frame_interval = max(1, video_fps // fps)
    
    frame_count = 0
    extracted_frame_count = 0
    
    while video_cap.isOpened() and extracted_frame_count < target_frame_count:
        ret, frame = video_cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            output_file_path = os.path.join(output_folder, f"{os.path.basename(video_path)}_frame_{extracted_frame_count:05d}.jpg")
            cv2.imwrite(output_file_path, frame)
            extracted_frame_count += 1
        
        frame_count += 1
    
    video_cap.release()
    return extracted_frame_count

def create_dataset_folders(base_folder):
    train_folder = os.path.join(base_folder, 'train')
    test_folder = os.path.join(base_folder, 'test')
    valid_folder = os.path.join(base_folder, 'val')
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)
    
    return train_folder, test_folder, valid_folder

def split_dataset(frame_folder, train_folder, test_folder, valid_folder, train_size=0.7, test_size=0.2, valid_size=0.1):
    frames = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg')]
    random.shuffle(frames)
    
    train_files, temp_files = train_test_split(frames, test_size=(1 - train_size))
    valid_files, test_files = train_test_split(temp_files, test_size=test_size/(test_size + valid_size))
    
    for file in train_files:
        os.rename(file, os.path.join(train_folder, os.path.basename(file)))
    
    for file in test_files:
        os.rename(file, os.path.join(test_folder, os.path.basename(file)))
    
    for file in valid_files:
        os.rename(file, os.path.join(valid_folder, os.path.basename(file)))

# Example usage
video_folder = r'C:\Users\Vedanta M S\Desktop\yolov8\iCOPEvid\Pain'
frame_folder = r'C:\Users\Vedanta M S\Desktop\yolov8\data'
dataset_base_folder = r'C:\Users\Vedanta M S\Desktop\yolov8\dataset_pain'

# Ensure frame folder exists
os.makedirs(frame_folder, exist_ok=True)

# Extract 100 frames from each of the first 10 videos in the folder
total_extracted_frames = 0
videos_processed = 0

for video_file in os.listdir(video_folder):
    if video_file.endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(video_folder, video_file)
        extracted_frames = extract_frames(video_path, frame_folder, target_frame_count=100)
        total_extracted_frames += extracted_frames
        videos_processed += 1
        if videos_processed >= 20:
            break

# Shuffle the extracted frames
frames = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg')]
random.shuffle(frames)

# Create dataset folders
train_folder, test_folder, valid_folder = create_dataset_folders(dataset_base_folder)

# Split frames into train, test, and validation sets
split_dataset(frame_folder, train_folder, test_folder, valid_folder)
