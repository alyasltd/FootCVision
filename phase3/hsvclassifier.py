import cv2
import numpy as np
import pandas as pd
import copy
from tqdm.auto import tqdm 
from typing import List
import supervision as sv
import sys
import os

# Get the absolute path of the parent directory (FootCVision)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase1.inference import PlayerInference

class hsvclassifier:
    def __init__(self, video_path, model_path="/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase1/runs/detect/train/weights/best.pt", conf_threshold=0.8, iou_threshold=0.8):
        """
        Initialize the class for extracting player crops from a video using YOLO detection.

        Args:
            video_path (str): Path to the input video.
            model_path (str): Path to the trained YOLO model weights.
            conf_threshold (float): Confidence threshold for detections.
            iou_threshold (float): IoU threshold for non-max suppression.
        """
        self.video_path = video_path
        self.detector = PlayerInference(model_path, conf_threshold, iou_threshold)  # Use PlayerInference
        # Defined color ranges (lower and upper HSV bounds) for different teams
        self.color_ranges = {
            "man_united": [(17, 0, 138), (122, 113, 255)],
            "liverpool": [(18, 0, 136), (129, 116, 255)]
        }
    
    def get_crops_from_frames(self, stride=30, player_id=2):
        """
        Extract player crops from video frames using PlayerInference.

        Args:
            stride (int): Number of frames to skip between processed frames.
            player_id (int): Class ID for players.

        Returns:
            List of cropped player images.
        """
        frame_generator = sv.get_video_frames_generator(
            source_path=self.video_path, stride=stride)

        crops = []
        for frame in tqdm(frame_generator, desc="Extracting player crops"):
            detections = self.detector.inference(frame)  # Use PlayerInference for detection

            # Apply Non-Maximum Suppression (NMS) and filter only player detections
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            players_detections = detections[detections.class_id == player_id]
            
            # Extract player crops
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            crops += players_crops  # Add crops to the list

        return crops


    def plot_crops(self, crops):
        """
        Display extracted player crops in a grid.

        :param crops: List of cropped player images.
        """
        if not crops:
            print("No crops to display.")
            return

        # Display all crops (up to 100) in a grid
        sv.plot_images_grid(crops[:100], grid_size=(10, 10))

    def get_hsv_img(self, img: np.ndarray) -> np.ndarray:
        """
        Convert the image to HSV color space
        """
        return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    def apply_filter(self, img: np.ndarray, filter_range: tuple) -> np.ndarray:
        """
        Apply a color filter to an image using a given HSV range.
        
        Parameters:
        - img: The input image to apply the filter.
        - filter_range: A tuple containing (lower_hsv, upper_hsv) to filter the image.
        
        Returns:
        - Filtered image with applied color mask.
        """
        lower_hsv, upper_hsv = filter_range
        img_hsv = self.get_hsv_img(img)
        mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        return cv2.bitwise_and(img, img, mask=mask)

    def crop_img_for_jersey(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image to focus on the player's jersey area based on the bounding box coordinates (x, y, w, h).
        
        Parameters:
        - img: The input image (frame).
        - x: The x-coordinate of the top-left corner of the bounding box.
        - y: The y-coordinate of the top-left corner of the bounding box.
        - w: The width of the bounding box.
        - h: The height of the bounding box.
        
        Returns:
        - Cropped image focusing on the jersey area of the player.
        """
        height, width, _ = img.shape

        y_start = int(height * 0.15)
        y_end = int(height * 0.50)
        x_start = int(width * 0.15)
        x_end = int(width * 0.85)

        return img[y_start:y_end, x_start:x_end]
    

    def add_median_blur(self, img: np.ndarray) -> np.ndarray:
        """
        Apply median blur to reduce noise in the image.
        """
        return cv2.medianBlur(img, 5)


    def crop_filter_and_blur_img(self, img: np.ndarray, filter_range: tuple) -> np.ndarray:
        """
        Crop the image, apply color filtering, and apply median blur.
        """
        transformed_img = self.crop_img_for_jersey(img)
        transformed_img = self.apply_filter(transformed_img, filter_range)
        transformed_img = self.add_median_blur(transformed_img)
        return transformed_img


    def predict_img(self, img: np.ndarray) -> str:
        """
        Predict the team name based on the color of the jersey.
        """
        if img is None:
            raise ValueError("Image can't be None")

        max_non_black_pixel_count = 0
        predicted_team = "Unknown"

        for team, filter_range in self.color_ranges.items():
            # Apply the color filter for each team
            filter_data = self.add_non_black_pixels_count_in_filter(img, filter_range)
            if filter_data["non_black_pixels_count"] > max_non_black_pixel_count:
                max_non_black_pixel_count = filter_data["non_black_pixels_count"]
                predicted_team = team

        return predicted_team

    def collect_and_classify(self) -> dict:
        """
        Collects the bounding box coordinates from df_tracking, extracts and processes jersey crops,
        applies filtering, blurring, and classifies each player's team.

        Returns:
            Dictionary with keys as `frame_number` and values as lists of processed crops.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return {}

        classified_crops = {}  # Dictionary to store classified crops per frame
        classified_data = []  # Store classification results

        for frame_number in tqdm(self.df_tracking["frame"].unique(), desc="Processing frames"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()

            if not ret:
                print(f"Warning: Could not read frame {frame_number}")
                continue

            # Filter tracking data for the current frame
            frame_data = self.df_tracking[self.df_tracking["frame"] == frame_number]

            crops = []  # Store processed crops for this frame

            for _, row in frame_data.iterrows():
                if row["class"] != "player":
                    continue  # Ignore non-player objects

                x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])

                # Ensure bounding box is valid
                if w <= 0 or h <= 0:
                    continue

                # Crop player's image
                player_crop = frame[y:y+h, x:x+w]

                # Crop only the jersey area
                jersey_crop = self.crop_img_for_jersey(player_crop)

                # Apply filter and blur
                processed_crop = self.crop_filter_and_blur_img(jersey_crop, self.color_ranges["man_united"])  # Default filter
                team = self.predict_img(jersey_crop)  # Predict team

                # Store classified crop
                crops.append(processed_crop)

                # Append classification results
                classified_data.append({
                    "frame": frame_number,
                    "track_id": row["track_id"],
                    "team": team,
                    "class": row["class"]
                })

            # Store crops for this frame
            classified_crops[frame_number] = crops

        cap.release()

        # Convert classification results to DataFrame
        classified_df = pd.DataFrame(classified_data)
        classified_df.to_csv('/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase3/classified.csv', index=False)

        return classified_crops


if __name__ == "__main__":
    # Load your tracking data
    df_tracking = pd.read_csv('/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase2/test.csv')

    # Specify your video path
    video_path = '/Users/alyazouzou/Desktop/CV_Football/vids/good.mov'

    # Initialize the classifier
    hsv_classifier = hsvclassifier(video_path, df_tracking)

    # Run the classification process
    classified_df = hsv_classifier.collect_and_classify(video_path, df_tracking)

    # Optionally: print the classified DataFrame and save it
    print(classified_df)
    classified_df.to_csv('/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase3/classified.csv', index=False)
