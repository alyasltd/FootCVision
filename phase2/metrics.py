import numpy as np
from typing import List
import supervision as sv

class Metrics:
    def __init__(self, fps=30, possession_threshold=20, ball_distance_threshold=100):
        """
        Tracks ball possession based on player distance to the ball.
        
        Args:
            fps (int): Frames per second of the video.
            possession_threshold (int): Number of consecutive frames a player must be closest to be considered in possession.
            ball_distance_threshold (int): Maximum distance to consider a player in possession of the ball.
        """
        self.fps = fps
        self.possession_threshold = possession_threshold
        self.ball_distance_threshold = ball_distance_threshold
        self.current_team = None  # Track which team has possession
        self.possession_counter = 0  # Track how long a team has possession
        self.closest_player = None  # Player closest to the ball

    def update_ball_poss(self, players: sv.Detections, ball: sv.Detections):
        """
        Updates the ball possession based on player distances.

        Args:
            players (sv.Detections): Players detected in the frame.
            ball (sv.Detections): Ball detected in the frame.
        """
        # 🚨 Debugging: Check if we detect anything
        print(f"🔍 Detected {len(players)} players, {len(ball)} ball(s)")

        if len(players) == 0 or len(ball) == 0:
            print("⚠️ No players or ball detected. Skipping possession update.")
            self.closest_player = None
            self.possession_counter = 0  # Reset possession if ball is missing
            return None  # No possession change

        # Extract coordinates
        ball_xy = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]  # Get ball center
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)  # Get player feet positions
        player_ids = players.tracker_id  # Get player tracking IDs
        player_teams = players.class_id  # Get player team assignments

        if len(players_xy) == 0 or len(player_ids) == 0:
            print("⚠️ No valid player detections found!")
            return None  # No valid player detections

        # Compute distances of all players to the ball
        distances = np.linalg.norm(players_xy - ball_xy, axis=1)
        closest_idx = np.argmin(distances)  # Get the index of the closest player

        if distances[closest_idx] > self.ball_distance_threshold:
            print(f"⚠️ Closest player is too far ({distances[closest_idx]:.2f}px). No possession assigned.")
            self.closest_player = None  # No one has possession
            self.possession_counter = 0  # Reset counter
            return None

        # Get closest player's info
        closest_player_id = player_ids[closest_idx]
        closest_team = player_teams[closest_idx]

        print(f"✅ Closest player: ID {closest_player_id}, Team {closest_team}, Distance {distances[closest_idx]:.2f}px")

        # Change possession only if held for consecutive frames
        if closest_team != self.current_team:
            print(f"⚠️ Team changed! Resetting possession counter. New team: {closest_team}")
            self.possession_counter = 0  # Reset counter when switching teams
            self.current_team = closest_team

        self.possession_counter += 1

        if self.possession_counter >= self.possession_threshold:
            print(f"🎯 Possession confirmed: Team {self.current_team}")
            return self.current_team  # Confirmed possession

        return None  # No change in possession yet

    def get_possession_status(self):
        """
        Returns possession status as a formatted string.
        """
        return f"⚽ Ball Possession: Team {self.current_team}" if self.current_team is not None else "⚠️ No Possession"