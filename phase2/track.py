import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import CubicSpline

class PlayerTracker:
    def __init__(self, video_path, model_path="/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase1/runs/detect/train/weights/best.pt"):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.track_history = defaultdict(lambda: [])
        self.ball_tracking_data = []
        self.player_tracking_data = []
        self.class_labels = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

    def track(self):
        """
        Track the players and the ball in the video and return a DataFrame with the tracking data.
        Args:
            self : object
        Returns:
            DataFrame: A DataFrame containing the tracking data.
        """

        cap = cv2.VideoCapture(self.video_path)
        frame_number = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1
            image_height, image_width, _ = frame.shape

            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, class_id in zip(boxes, track_ids, classes):
                x, y, w, h = box
                #x1, y1, x2, y2 = x / image_width, y / image_height, (x + w) / image_width, (y + h) / image_height

                track = self.track_history[track_id]
                #print(track)
                track.append((float(x), float(y), float(w), float(h)))

                if len(track) > 30:
                    track.pop(0)

                if class_id == 0:  # "ball" class
                    self.ball_tracking_data.append({
                        "frame": frame_number,
                        "track_id": track_id,
                        "class": "ball",
                        "x": float(x),
                        "y": float(y),
                        "w": float(w),
                        "h": float(h)
                    })
                elif class_id == 2:  # "player" class
                    self.player_tracking_data.append({
                        "frame": frame_number,
                        "track_id": track_id,
                        "class": "player",
                        "x": float(x),
                        "y": float(y),
                        "w": float(w),
                        "h": float(h)
                    })

        cap.release()

        ball_df = pd.DataFrame(self.ball_tracking_data)
        player_df = pd.DataFrame(self.player_tracking_data)
        #ball_df.to_csv('ball.csv')
        #player_df.to_csv('player.csv')

        #ball_df_full = self._interpolate_and_fill(ball_df)
        #player_df_full = self._interpolate_and_fill(player_df)
        #ball_df_full.to_csv('ball_full.csv')
        #player_df_full.to_csv('player_full.csv')   

        final_df = pd.concat([ball_df, player_df])
        final_df = final_df.sort_values(by=['frame', 'track_id']).reset_index(drop=True)
        final_df['track_id'] = final_df['track_id'].astype(int)

        return final_df



    def _interpolate_and_fill(self, df):
        """
        Interpolate missing values in the DataFrame and fill the remaining missing values.
        Args:
            df (DataFrame): The input DataFrame.
        Returns:
            DataFrame: The DataFrame with missing values interpolated and filled.
        """
        if df.empty:
            return df

        min_frame = df['frame'].min()
        max_frame = df['frame'].max()
        all_frames = pd.DataFrame({'frame': range(min_frame, max_frame + 1)})
        df_full = pd.merge(all_frames, df, on='frame', how='left')

        for col in ['x', 'y', 'w', 'h']:
            missing = df_full[col].isna()
            df_training = df_full[~missing]
            df_missing = df_full[missing].reset_index(drop=True)

            if not df_training.empty:
                f = interp1d(df_training['frame'], df_training[col], fill_value="extrapolate")
                df_full.loc[missing, col] = f(df_missing['frame'])

        df_full['class'] = df_full['class'].ffill()
        missing_track_id = df_full['track_id'].isna()

        for idx in df_full[missing_track_id].index:
            if idx == 0:
                df_full.loc[idx, 'track_id'] = 1
            else:
                df_full.loc[idx, 'track_id'] = df_full.loc[idx - 1, 'track_id'] + 1

        return df_full



    
    def plot_tracking(self, final_df):
        """
        Plot the tracking data on the video.
        Args:
            final_df (DataFrame): The DataFrame containing the tracking data.
        """
        cap = cv2.VideoCapture(self.video_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_number = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1

            # Get the tracking data for the current frame
            frame_data = final_df[final_df['frame'] == frame_number]

            for _, row in frame_data.iterrows():
                x_center = row['x'] 
                y_center = row['y'] 
                box_width = row['w'] 
                box_height = row['h'] 
                class_name = row['class']

                # Calculate top-left and bottom-right coordinates of the bounding box
                x_min = int(x_center - box_width / 2)
                y_min = int(y_center - box_height / 2)
                x_max = int(x_center + box_width / 2)
                y_max = int(y_center + box_height / 2)

                # Choose a color for the bounding box
                color = (0, 255, 0) if class_name == "ball" else (255, 0, 0)

                # Draw the bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Write the frame to the output video
            out.write(frame)

        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()

# Usage example
#video_path = "/Users/alyazouzou/Desktop/CV_Football/vids/mcchelsea.mov"
#tracker = PlayerTracker(video_path)
#final_df = tracker.track()
#print(final_df)