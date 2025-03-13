import numpy as np
import cv2
import supervision as sv
from tqdm import tqdm
import sys
import os
# Get the absolute path of the parent directory (FootCVision)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from phase1.inference import PlayerInference
from phase3.hsvclassifier import HSVClassifier  # Import the HSV classifier

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

        # ✅ Initialize HSVClassifier as an object
        self.hsv_classifier = HSVClassifier(video_path, model_path, conf_threshold, iou_threshold)

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
            
            # Extract and track players
            detections.xyxy = sv.pad_boxes(detections.xyxy, px=10, py=10)
            all_detections = detections[detections.class_id == 2]  # Only players
            all_detections = self.tracker.update_with_detections(detections=all_detections)

            # Extract crops and classify using HSVClassifier
            print(f"🧐 Detected {len(all_detections)} players")
            player_crops = [sv.crop_image(frame, xyxy) for xyxy in all_detections.xyxy]
            print(f"📸 Extracted {len(player_crops)} player crops for HSV classification")

            if player_crops:
                predicted_classes = [self.hsv_classifier.predict_team(crop) for crop in player_crops]
                
                # ✅ Convert team names to numeric IDs
                predicted_classes_numeric = np.array([team_mapping.get(team, -1) for team in predicted_classes])  
                all_detections.class_id = predicted_classes_numeric

            # Annotate frame
            annotated_frame = self.annotate_frame(frame, all_detections)

            # ✅ Plot only the 8th frame
            if frame_count == 8:
                print("📸 Displaying Frame 8")
                sv.plot_image(annotated_frame)
                break  # Stop the loop after displaying the 8th frame

        if save_video:
            out.release()
        cv2.destroyAllWindows()

    def annotate_frame(self, frame, all_detections):
        """
        Annotates the frame with classified player detections.

        Args:
            frame (np.ndarray): The current video frame.
            all_detections (sv.Detections): Tracked player detections.
        """
        labels = [f"Team {team_id}" for team_id in all_detections.class_id]

        ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493']), thickness=2)
        label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493']),
                                            text_color=sv.Color.from_hex('#000000'),
                                            text_position=sv.Position.BOTTOM_CENTER)

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=all_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=all_detections, labels=labels)
        sv.plot_image(annotated_frame)
        return annotated_frame