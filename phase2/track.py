import numpy as np
import supervision as sv
import sys
import os
import time
# Get the absolute path of the parent directory (FootCVision)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ultralytics import YOLO
from tqdm import tqdm
import cv2
from phase1.inference import PlayerInference
from phase3.kmeansclassifier import kmeansclassifier  # Assuming KMeans is used for team classification

class Track:
    def __init__(self, video_path, model_path="/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase1/runs/detect/train/weights/best.pt", stride=30, conf_threshold=0.3, iou_threshold=0.5):
        """
        Handles player tracking, classification, and goalkeeper assignment.

        Args:
            video_path (str): Path to the input football match video.
            model_path (str): Path to the trained YOLO model weights.
            stride (int): Frame skipping interval for faster processing.
            conf_threshold (float): Confidence threshold for YOLO detections.
            iou_threshold (float): IoU threshold for non-max suppression.
        """
        self.video_path = video_path
        self.detector = PlayerInference(model_path, conf_threshold, iou_threshold)
        self.kmeans_classifier = kmeansclassifier(video_path, n_clusters=2)  # Initialize KMeans classifier
        self.tracker = sv.ByteTrack()  
        self.tracker.reset()

    def assign_goalkeeper_to_team(self, players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
        """
        Assigns goalkeepers to the closest team based on player positions.

        Args:
            players (sv.Detections): Detected outfield players with known team assignments.
            goalkeepers (sv.Detections): Detected goalkeepers without team assignments.

        Returns:
            np.ndarray: Assigned team IDs for goalkeepers.
        """
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

        # Ensure players.class_id is a NumPy array
        player_classes = np.array(players.class_id).flatten()

        # Ensure players exist before computing centroids
        if len(players_xy) == 0 or len(goalkeepers_xy) == 0:
            return np.zeros(len(goalkeepers_xy), dtype=int)  # Default all GKs to team 0

        # Filter team players correctly
        team_0_players = players_xy[np.where(player_classes == 0)] if np.any(player_classes == 0) else np.empty((0, 2))
        team_1_players = players_xy[np.where(player_classes == 1)] if np.any(player_classes == 1) else np.empty((0, 2))

        # Handle cases where a team has no detected players
        if team_0_players.shape[0] == 0 or team_1_players.shape[0] == 0:
            return np.zeros(len(goalkeepers_xy), dtype=int)  # Assign all GKs to one team if issue

        # Compute team centers using median for robustness
        team_0_center = np.median(team_0_players, axis=0) if team_0_players.shape[0] > 0 else np.array([0, 0])
        team_1_center = np.median(team_1_players, axis=0) if team_1_players.shape[0] > 0 else np.array([0, 0])

        gk_team_assignments = []
        for gk_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(gk_xy - team_0_center)
            dist_1 = np.linalg.norm(gk_xy - team_1_center)
            gk_team_assignments.append(0 if dist_0 < dist_1 else 1)

        return np.array(gk_team_assignments, dtype=int)

    def track_and_classify(self, save_video=True, output_path="output.mp4"):
        """
        Tracks, classifies, and assigns players, goalkeepers, referees, and the ball.
        Optionally saves the annotated frames as a video.
        """
        frame_generator = sv.get_video_frames_generator(source_path=self.video_path, stride=1)
        first_frame = next(frame_generator)
        height, width, _ = first_frame.shape
        cap = cv2.VideoCapture(self.video_path) 
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()  # Close video file


        # Set up video writer
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 codec
            out = cv2.VideoWriter(output_path, fourcc, int(fps), (width, height))  # 30 FPS

        # Step 1: Collect training crops from multiple frames BEFORE tracking
        if not hasattr(self.kmeans_classifier, "trained") or not self.kmeans_classifier.trained:
            print("🔄 Collecting training data for KMeans...")
            training_crops = self.kmeans_classifier.get_crops_from_frames(stride=150, player_id=2) # Extract training crops

            if len(training_crops) > 0:
                training_features = self.kmeans_classifier.get_features(training_crops)  # Extract features
                self.kmeans_classifier.train_kmeans(training_features)  # Train KMeans
                self.kmeans_classifier.trained = True  # Mark as trained
            else:
                print("⚠️ No training data available for KMeans!")

        for frame in tqdm(frame_generator, desc="Processing frames"):

            detections = self.detector.inference(frame)
            
            #print(detections)
            # Extract and process detections
            ball_detections = detections[detections.class_id == 0]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
            
            detections.xyxy = sv.pad_boxes(detections.xyxy, px=10, py=10)
            all_detections = detections[detections.class_id != 0]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
            all_detections = self.tracker.update_with_detections(detections=all_detections)
            

            goalkeepers = all_detections[all_detections.class_id == 1]
            players = all_detections[all_detections.class_id == 2]
            #print("here players",players)
            referees = all_detections[all_detections.class_id == 3]

            # Extract CLIP embeddings for player crops
            print(f"🧐 Detected {len(players)} players")
            player_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
            print(f"📸 Extracted {len(player_crops)} player crops for classification")
            player_features = self.kmeans_classifier.get_features(player_crops)

            # Step 2: Classify players into two teams using KMeans
            if len(player_features) > 0:
                print(f"🧐 Players BEFORE KMeans: {len(players)}")
                # Predict cluster labels (0 or 1) for players
                predicted_classes = self.kmeans_classifier.predict_clusters(player_features)

                if len(predicted_classes) < len(players):
                    print(f"⚠️ Mismatch: {len(players)} players but {len(predicted_classes)} predicted classes!")
                    
                    # Fill missing class IDs with -1 (or any default class)
                    missing_count = len(players) - len(predicted_classes)
                    predicted_classes = np.concatenate([predicted_classes, np.full(missing_count, -1)])


                # Assign corrected class IDs (Shift by 2 so they don’t overlap with ball and goalkeeper)
                players.class_id = np.array(predicted_classes).flatten() + 2  # Shift by 2
                print(f"🎯 Players AFTER KMeans Classification: {len(players)}")

            # Assign goalkeepers to teams
            goalkeepers.class_id = self.assign_goalkeeper_to_team(players, goalkeepers)

            # Merge detections
            all_detections = sv.Detections.merge([players, goalkeepers, referees])
            annotated_frame = self.annotate_frame(frame, all_detections, ball_detections)

            # Write to video file
            if save_video==True:
                out.write(annotated_frame)
                cv2.imshow('frame', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if save_video:
            out.release()
        cv2.destroyAllWindows()

    def annotate_frame(self, frame, all_detections, ball_detections):
        """
        Annotates the frame with player, goalkeeper, referee, and ball detections.

        Args:
            frame (np.ndarray): The current video frame.
            all_detections (sv.Detections): Merged detections of players, goalkeepers, and referees.
            ball_detections (sv.Detections): Detected ball positions.
        """
        labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]

        ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']), thickness=2)
        label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
                                            text_color=sv.Color.from_hex('#000000'),
                                            text_position=sv.Position.BOTTOM_CENTER)
        triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex('#FFD700'), base=25, height=21, outline_thickness=1)

        # Apply annotations
        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=all_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=all_detections, labels=labels)
        annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)
        
        sv.plot_image(annotated_frame)
        return annotated_frame
    
    def ball_possession(self, ball_detections, team_0_players, team_1_players):
        """
        Determines which team has possession of the ball based on the ball's position.

        Args:
            ball_detections (sv.Detections): Detected ball positions.
            team_0_players (sv.Detections): Detected players from team 0.
            team_1_players (sv.Detections): Detected players from team 1.

        Returns:
            int: Team ID (0 or 1) that has possession of the ball.
        """
        pass
        

if __name__ == "__main__":
    video_path = "/Users/alyazouzou/Desktop/CV_Football/vids/good.mov"
    tracker = Track(video_path)
    tracker.track_and_classify(save_video=False)
    