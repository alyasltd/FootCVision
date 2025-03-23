import numpy as np
import cv2
import supervision as sv
from tqdm import tqdm
import sys
from metrics import Metrics
import os
# Get the absolute path of the parent directory (FootCVision)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from phase1.inference import PlayerInference
from phase2.hsvclassifier import HSVClassifier  # Import the HSV classifier

class TrackHSV:
    def __init__(self, video_path, model_path="/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase1/runs/detect/train/weights/best.pt", stride=30, conf_threshold=0.3, iou_threshold=0.5):
        """
        Handles player tracking and HSV-based classification.

        Args:
            video_path (str): Path to the input football match video.
            model_path (str): Path to the trained YOLO model weights.
            stride (int): Frame skipping interval for faster processing.
            conf_threshold (float): Confidence threshold for YOLO detections.
            iou_threshold (float): IoU threshold for non-max suppression.
        """
        self.video_path = video_path
        self.detector = PlayerInference(model_path, conf_threshold, iou_threshold)  # YOLO for detection
        self.tracker = sv.ByteTrack()
        self.tracker.reset()
        self.hsv_classifier = HSVClassifier(video_path, model_path, conf_threshold, iou_threshold)
        

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

    def track_and_classify_hsv(self, save_video=True, output_path="output_hsv.mp4"):
        """
        Tracks and classifies players using HSV classification.

        Args:
            save_video (bool): Whether to save the output video.
            output_path (str): File path for the saved video.
        """
        frame_generator = sv.get_video_frames_generator(self.video_path, stride=1)
        first_frame = next(frame_generator)
        height, width, _ = first_frame.shape
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.metrics = Metrics(fps=fps, possession_threshold=40, ball_distance_threshold=150)
        self.metrics.current_team = 0  # Initialize possession to team 0
        
        cap.release()

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(output_path, fourcc, int(fps), (width, height))

        # Define a mapping for team names to numeric class IDs
        team_mapping = {
            "1": 1, # "manchester_united": 1
            "0": 0  # "liverpool": 0
        }

        frame_count = 0  # Initialize frame counter

        for frame in tqdm(frame_generator, desc="Processing frames"):
            frame_count += 1  # Increment frame counter

            detections = self.detector.inference(frame)

            ball_detections = detections[detections.class_id == 0]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
            
            # Extract and track players
            detections.xyxy = sv.pad_boxes(detections.xyxy, px=10, py=10)
            all_detections = detections[detections.class_id == 2]  # Filter players only
            all_detections = self.tracker.update_with_detections(detections=all_detections)

        
            #print(f"🔍 Tracked {len(all_detections)} players")
            #print(f"🔍 Tracked {(all_detections)} players")

            # Extract crops and classify using HSVClassifier
            #print(f"🧐 Detected {len(all_detections)} players")
            player_crops = [sv.crop_image(frame, xyxy) for xyxy in all_detections.xyxy]
            #print(f"📸 Extracted {len(player_crops)} player crops for HSV classification")

            if player_crops:
                # ✅ Only process non-empty crops
                valid_crops = [crop for crop in player_crops if crop is not None and crop.size > 0]

                if valid_crops:
                    predicted_classes = [self.hsv_classifier.predict_team(crop) for crop in player_crops]
    
                    valid_indices = [i for i, team in enumerate(predicted_classes) if team in team_mapping]

                    all_detections = all_detections[valid_indices]  # Remove unclassified players
                    predicted_classes_numeric = np.array([team_mapping[predicted_classes[i]] for i in valid_indices])
                    
                    all_detections.class_id = predicted_classes_numeric  # Update class IDs with numeric values

            possession_team = self.metrics.update_ball_poss(all_detections, ball_detections)

            #possession_player_id = self.metrics.current_player_id
            #print(possession_player_id)
            
            #print("\n📍 Player Vectors:")
            #for i in range(len(all_detections)):
            #    tracker_id = all_detections.tracker_id[i]
            #    class_id = all_detections.class_id[i]
            #    bbox = all_detections.xyxy[i]
            #    bottom_center = all_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[i]

            #    print(f"🧍 Player ID {tracker_id} | Team {class_id} | BBox: {bbox} | Bottom-Center: {bottom_center}")


            # Annotate frame
            annotated_frame = self.annotate_frame(frame, all_detections, ball_detections)
            #sv.plot_image(annotated_frame)
            
            #if frame_count==3:
            #   break
                #sv.plot_image(annotated_frame)

            if save_video==True:
                out.write(annotated_frame)

        if save_video == True:
            out.release()
        cv2.destroyAllWindows()

    def annotate_frame(self, frame, all_detections, ball_detections):
        """
        Annotates the frame with classified player detections.

        Args:
            frame (np.ndarray): The current video frame.
            all_detections (sv.Detections): Tracked player detections.
        """
        possession_player_id = self.metrics.current_player_id
        labels = [
        f"ID {tracker_id} - Team {team_id}" 
        for tracker_id, team_id in zip(all_detections.tracker_id, all_detections.class_id)]
        ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493']), thickness=2)
        label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493']),
                                            text_color=sv.Color.from_hex('#000000'),
                                            text_position=sv.Position.BOTTOM_CENTER)
        triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex('#FFD700'), base=25, height=21, outline_thickness=1)


        annotated_frame = frame.copy()
        for i, tracker_id in enumerate(all_detections.tracker_id):
            if tracker_id == possession_player_id:
                foot_xy = all_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[i]
                cv2.circle(annotated_frame, tuple(foot_xy.astype(int)), 15, (0, 255, 0), 3)  # Green circle
                
        annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=all_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=all_detections, labels=labels)
        annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)
        
        # ✅ Add Ball Possession Text Overlay
        possession_team = self.metrics.current_team
        possession_text = f"Ball Possession: Team {possession_team}" if possession_team is not None else "No Possession"

        cv2.putText(
            annotated_frame, possession_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2, cv2.LINE_AA  # Green text for visibility
        )
        
        return annotated_frame
        #sv.plot_image(annotated_frame)