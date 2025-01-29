import pandas as pd
import numpy as np
from ultralytics import YOLO
import cv2
from collections import defaultdict

class PlayerTracker:
    def __init__(self, model_path="/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase1/runs/detect/train/weights/best.pt", tracker="botsort.yaml"):
        self.model = YOLO(model_path)  # Load the YOLO model
        self.players = {}  # Dictionary to store player IDs and their corresponding data
        self.ball = None  # Placeholder for the ball position
        self.goalkeepers = {}  # Store goalkeeper data
        self.referees = {}  # Store referee data
        self.frames = []  # Store frames for interpolation
        self.player_positions = defaultdict(list)  # Track player positions over time
        self.ball_positions = []  # Track ball positions over time
        self.goalkeeper_positions = defaultdict(list)  # Track goalkeeper positions over time
        self.referee_positions = defaultdict(list)  # Track referee positions over time

    def track(self, frame):
        """
        Track players, ball, goalkeepers, and referees in a sequence of frames using YOLO model.
        """
        # Get YOLO detections for the current frame
        results = self.model.track(frame, persist=True)  # Get YOLO detections
        detections = results.pandas().xywh  # Get the bounding box coordinates from YOLO
        
        # Store current frame results
        self.frames.append(frame)

        # Reset the detection lists for the current frame
        self.players.clear()
        self.ball = None
        self.goalkeepers.clear()
        self.referees.clear()

        # Process the detections for players, ball, referees, and goalkeepers
        for _, row in detections.iterrows():
            if row['name'] == 'player':  # Track players
                self.players[row['id']] = {'bbox': row[['xmin', 'ymin', 'xmax', 'ymax']].values}
                self.player_positions[row['id']].append(((row['xmin'] + row['xmax']) / 2, (row['ymin'] + row['ymax']) / 2))
            elif row['name'] == 'ball':  # Track the ball
                self.ball = {'bbox': row[['xmin', 'ymin', 'xmax', 'ymax']].values}
                self.ball_positions.append(((row['xmin'] + row['xmax']) / 2, (row['ymin'] + row['ymax']) / 2))
            elif row['name'] == 'referee':  # Track referees
                self.referees[row['id']] = {'bbox': row[['xmin', 'ymin', 'xmax', 'ymax']].values}
                self.referee_positions[row['id']].append(((row['xmin'] + row['xmax']) / 2, (row['ymin'] + row['ymax']) / 2))
            elif row['name'] == 'goalkeeper':  # Track goalkeepers
                self.goalkeepers[row['id']] = {'bbox': row[['xmin', 'ymin', 'xmax', 'ymax']].values}
                self.goalkeeper_positions[row['id']].append(((row['xmin'] + row['xmax']) / 2, (row['ymin'] + row['ymax']) / 2))

        # Perform interpolation on player, ball, goalkeeper, and referee positions
        self.interpolate_positions()

        return self.players, self.ball, self.goalkeepers, self.referees

    def interpolate_positions(self):
        """
        Interpolate missing player, ball, referee, and goalkeeper positions over time.
        """
        self.interpolate_player_positions()
        self.interpolate_ball_positions()
        self.interpolate_goalkeeper_positions()
        self.interpolate_referee_positions()

    def interpolate_player_positions(self):
        """
        Interpolate missing player positions using pandas.
        """
        for player_id, positions in self.player_positions.items():
            if len(positions) > 1:  # Only interpolate if more than one position is available
                df_positions = pd.DataFrame(positions, columns=['x', 'y'])
                df_positions = df_positions.interpolate(method='linear')  # Linear interpolation
                df_positions = df_positions.bfill()  # Backfill remaining missing values
                self.player_positions[player_id] = df_positions.values.tolist()

    def interpolate_ball_positions(self):
        """
        Interpolate missing ball positions using pandas.
        """
        if self.ball_positions:
            df_ball_positions = pd.DataFrame(self.ball_positions, columns=['x', 'y'])
            df_ball_positions = df_ball_positions.interpolate(method='linear')  # Linear interpolation
            df_ball_positions = df_ball_positions.bfill()  # Backfill remaining missing values
            self.ball_positions = df_ball_positions.values.tolist()

    def interpolate_goalkeeper_positions(self):
        """
        Interpolate missing goalkeeper positions using pandas.
        """
        for goalkeeper_id, positions in self.goalkeeper_positions.items():
            if len(positions) > 1:  # Only interpolate if more than one position is available
                df_positions = pd.DataFrame(positions, columns=['x', 'y'])
                df_positions = df_positions.interpolate(method='linear')  # Linear interpolation
                df_positions = df_positions.bfill()  # Backfill remaining missing values
                self.goalkeeper_positions[goalkeeper_id] = df_positions.values.tolist()

    def interpolate_referee_positions(self):
        """
        Interpolate missing referee positions using pandas.
        """
        for referee_id, positions in self.referee_positions.items():
            if len(positions) > 1:  # Only interpolate if more than one position is available
                df_positions = pd.DataFrame(positions, columns=['x', 'y'])
                df_positions = df_positions.interpolate(method='linear')  # Linear interpolation
                df_positions = df_positions.bfill()  # Backfill remaining missing values
                self.referee_positions[referee_id] = df_positions.values.tolist()

