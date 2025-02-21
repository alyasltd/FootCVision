import cv2
import numpy as np
import pandas as pd
import copy
from typing import List

class hsvclassifier:
    def __init__(self, video_path, df_tracking):
        self.video_path = video_path
        self.df_tracking = df_tracking

        # Defined color ranges (lower and upper HSV bounds) for different teams
        self.color_ranges = {
            "man_united": [(17, 0, 138), (122, 113, 255)],
            "liverpool": [(18, 0, 136), (129, 116, 255)]
        }

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
        Crop the image to focus on the player's jersey area.
        """
        height, width, _ = img.shape

        y_start = int(height * 0.15)
        y_end = int(height * 0.50)
        x_start = int(width * 0.15)
        x_end = int(width * 0.85)

        return img[y_start:y_end, x_start:x_end]
    
    def crop_img_for_jersey(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
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
        return img[y:y+h, x:x+w]


    def add_median_blur(self, img: np.ndarray) -> np.ndarray:
        """
        Apply median blur to reduce noise in the image.
        """
        return cv2.medianBlur(img, 5)

    def non_black_pixels_count(self, img: np.ndarray) -> float:
        """
        Count the number of non-black pixels in the image.
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.countNonZero(img_gray)

    def crop_filter_and_blur_img(self, img: np.ndarray, filter_range: tuple) -> np.ndarray:
        """
        Crop the image, apply color filtering, and apply median blur.
        """
        transformed_img = self.crop_img_for_jersey(img)
        transformed_img = self.apply_filter(transformed_img, filter_range)
        transformed_img = self.add_median_blur(transformed_img)
        return transformed_img

    def add_non_black_pixels_count_in_filter(self, img: np.ndarray, filter_range: tuple) -> dict:
        """
        Apply the filter to the image and count the non-black pixels.
        """
        transformed_img = self.crop_filter_and_blur_img(img, filter_range)
        non_black_pixel_count = self.non_black_pixels_count(transformed_img)
        return {"non_black_pixels_count": non_black_pixel_count}

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

    def collect_and_classify(self, video_path, df_tracking):
        """
        Collect the bounding box coordinates from df_tracking and classify the team based on the color of the cropped image.
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        classified_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # For each row in df_tracking (assuming it has columns: 'frame', 'track_id', 'x', 'y', 'w', 'h')
            for _, row in df_tracking[df_tracking['frame'] == int(cap.get(cv2.CAP_PROP_POS_FRAMES))].iterrows():
                # Extract bounding box and crop the image
                x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
                cropped_image = self.crop_img_for_jersey(frame)

                # Classify the team based on jersey color
                team = self.predict_img(cropped_image)

                classified_data.append({
                    "frame": row['frame'],
                    "track_id": row['track_id'],
                    "team": team,
                    "class": row['class'],
                    })

        cap.release()

        # Convert classified data into a DataFrame and return
        return pd.DataFrame(classified_data)

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
