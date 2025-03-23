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
        self.current_team = None
        self.possession_counter = 0
        self.closest_player = None
        self.current_player_id = None  # ✅ New: Track player ID with ball

        self.team_possession = {}
        self.total_frames = 0

    def update_ball_poss(self, players: sv.Detections, ball: sv.Detections):
        """"
        Updates the ball possession based on the closest player to the ball.
        """
        self.total_frames += 1

        if len(players) == 0 or len(ball) == 0:
            self.closest_player = None
            self.current_player_id = None
            self.possession_counter = 0
            return None

        ball_xy = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        player_ids = players.tracker_id
        player_teams = players.class_id

        if len(players_xy) == 0 or len(player_ids) == 0:
            return None

        distances = np.linalg.norm(players_xy - ball_xy, axis=1)
        closest_idx = np.argmin(distances)

        if distances[closest_idx] > self.ball_distance_threshold:
            self.closest_player = None
            self.current_player_id = None
            self.possession_counter = 0
            return None

        closest_team = player_teams[closest_idx]
        self.current_player_id = player_ids[closest_idx]  # ✅ Save player in possession

        if closest_team != self.current_team:
            self.possession_counter = 0
            self.current_team = closest_team

        self.possession_counter += 1

        if self.possession_counter >= self.possession_threshold:
            if closest_team not in self.team_possession:
                self.team_possession[closest_team] = 0
            self.team_possession[closest_team] += 1
            return self.current_team

        return None

    def get_possession_percentage(self):
        """"
        Computes the possession percentage for each team.
        """
        if self.total_frames == 0:
            return "Pas encore de données"

        possession_percentages = {
            team: (frames / self.total_frames) * 100
            for team, frames in self.team_possession.items()
        }

        return possession_percentages