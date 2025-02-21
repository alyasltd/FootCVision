import cv2
import numpy as np
import pandas as pd
import supervision as sv  # Assuming you're using supervision for visualization
from tqdm.auto import tqdm  # Correct tqdm import for Jupyter & scripts


class kmeansclassifier:
    def __init__(self, video_path, df_tracking):
        self.video_path = video_path
        self.df_tracking = df_tracking

    def get_crops_from_frames(self, target_frames):
        """
        Extract crops from multiple specified frames using tracking data.
        
        :param target_frames: List of frame numbers from which to extract crops.
        :return: Dictionary {frame_number: list_of_crops} containing cropped player images.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return {}

        all_crops = {}

        # Get video frame dimensions
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read video.")
            cap.release()
            return {}

        for frame_idx in tqdm(target_frames, desc="Extracting crops"):
            frame_data = self.df_tracking[self.df_tracking["frame"] == frame_idx]
            if frame_data.empty:
                print(f"No tracking data found for frame {frame_idx}")
                continue

            # Set video to the target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame {frame_idx}")
                continue

            crops = []
            for _, row in frame_data.iterrows():
                if row["class"] == "player":  # Only consider player crops
                    x_center = row["x"]
                    y_center = row["y"]
                    box_width = row["w"]
                    box_height = row["h"]

                    # Calculate top-left and bottom-right coordinates of the bounding box
                    x_min = int(x_center - box_width / 2)
                    y_min = int(y_center - box_height / 2)
                    x_max = int(x_center + box_width / 2)
                    y_max = int(y_center + box_height / 2)

                    # Ensure crop dimensions are valid
                    if x_max > x_min and y_max > y_min:
                        crop = frame[y_min:y_max, x_min:x_max]
                        if crop.size > 0:  # Ensure crop is not empty
                            crops.append(crop)

            all_crops[frame_idx] = crops  # Store crops for the frame

        cap.release()
        return all_crops


    def plot_crops(self, crops_dict):
        """
        Display all crops from multiple frames in a grid.
        
        :param crops_dict: Dictionary {frame_number: list_of_crops}
        """
        all_crops = []  # List to store all crops

        # Extract crops from all frames
        for frame_idx, crops in crops_dict.items():
            all_crops.extend(crops)  # Add crops from each frame to the list

        if not all_crops:
            print("No crops to display.")
            return

        # Display all crops (up to 100) in a grid
        sv.plot_images_grid(all_crops[:100], grid_size=(10, 10))

    def get_features():
        
        pass

    def projection_umap():
        pass

    def plot_projection():
        pass

video_path = "/Users/alyazouzou/Desktop/CV_Football/vids/good.mov"
df_tracking = pd.read_csv("/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase2/test.csv")
classifier = kmeansclassifier(video_path, df_tracking )
selected_frames = [1, 5, 10, 20, 50, 18, 30, 64, 36, 89]  # Example list of frames
crops = classifier.get_crops_from_frames(selected_frames)

if crops:
    classifier.plot_crops(crops) 
