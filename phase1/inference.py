from ultralytics import YOLO
import numpy as np
import supervision as sv

class PlayerInference:
    def __init__(self, model_path="/Users/alyazouzou/Desktop/CV_Football/FootCVision2/phase1/runs/detect/train/weights/best.pt", conf_threshold=0.3, iou_threshold=0.5):
        """
        Initialize the YOLO model for player detection.
        Args:
            model_path (str): Path to the trained YOLO model weights.
            conf_threshold (float): Confidence threshold for detections.
            iou_threshold (float): IoU threshold for non-max suppression.
        """
        self.model = YOLO(model_path)  # Load the YOLO model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def inference(self, frame):
        """
        Perform player detection on a single frame.
        Args:
            frame (np.ndarray): Input frame for player detection.
        Returns:
            sv.Detections: Supervision detection object containing bounding boxes, confidence scores, and class IDs.
        """
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)

        # Extract YOLO detections
        detections = []
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()  # Convert tensor to numpy array
                confidence = float(box.conf[0].cpu().numpy())  # Convert tensor to float
                class_id = int(box.cls[0].cpu().numpy())  # Convert tensor to int

                detections.append({"bbox": bbox, "confidence": confidence, "class_id": class_id})

        # Convert detections to Supervision format
        if detections:
            xyxy = np.array([det["bbox"] for det in detections])  # Bounding boxes
            confidence = np.array([det["confidence"] for det in detections])  # Confidence scores
            class_id = np.array([det["class_id"] for det in detections])  # Class IDs

            supervision_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            return supervision_detections
        else:
            return sv.Detections.empty()  # Return empty detections if none are found
        

    def track(self, frame, persist=True, tracker="/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase2/bytetrack.yaml"):
        """
        Perform player detection and tracking on a single frame.
        
        Args:
            frame (np.ndarray): Input frame for player detection.
            persist (bool): Whether to persist tracking across frames.

        Returns:
            sv.Detections: Supervision detection object containing bounding boxes, confidence scores, 
                           class IDs, and tracker IDs.
        """
        # Run YOLO tracking (ByteTrack by default)
        results = self.model.track(frame, conf=self.conf_threshold, iou=self.iou_threshold, persist=persist, tracker=tracker)

        # Extract YOLO detections
        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue  # Skip frames with no detections
            
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                confidence = float(box.conf[0].cpu().numpy())  # Confidence score
                class_id = int(box.cls[0].cpu().numpy())  # Object class
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1  # Tracking ID

                detections.append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "class_id": class_id,
                    "track_id": track_id
                })

        # Convert detections to Supervision format
        if detections:
            xyxy = np.array([det["bbox"] for det in detections])  # Bounding boxes
            confidence = np.array([det["confidence"] for det in detections])  # Confidence scores
            class_id = np.array([det["class_id"] for det in detections])  # Class IDs
            tracker_id = np.array([det["track_id"] for det in detections])  # Track IDs

            supervision_detections = sv.Detections(
                xyxy=xyxy, confidence=confidence, class_id=class_id, tracker_id=tracker_id
            )
            return supervision_detections
        else:
            return sv.Detections.empty()  # Return empty detections if none are found
        
    def detect(self, image_path, save_output=True):
        """
        Perform player detection on a single image.
        Args:
            image_path (str): Path to the input image.
            save_output (bool): If True, saves the detection visualization to disk.
        Returns:
            List[Dict]: Detected bounding boxes with confidence scores.
        """
        results = self.model(image_path, conf=self.conf_threshold, iou=self.iou_threshold)

        # Process results
        detections = []
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                confidence = box.conf[0].item()
                class_id = box.cls[0].item()
                detections.append({
                    "bbox": [int(coord) for coord in bbox],
                    "confidence": confidence,
                    "class_id": int(class_id),
                })

            if save_output:
                result.save()

        return detections
    
    def extract_boxes(self, image_paths):
        """
        Extract all bounding boxes from a list of images.
        Args:
            image_paths (List[str]): Paths to input images.
        Returns:
            List[np.ndarray]: List of bounding boxes for each image.
        """
        results = self.model(image_paths, conf=self.conf_threshold, iou=self.iou_threshold)
        all_boxes = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes (x1, y1, x2, y2)
            all_boxes.append(boxes)
        return all_boxes

    def extract_boxes_in_batches(self, image_paths, batch_size=8):
      """
      Extract bounding boxes from images in smaller batches to avoid GPU memory issues.
      Args:
          image_paths (List[str]): List of image paths to process.
          batch_size (int): Number of images to process per batch.
      Returns:
          List[np.ndarray]: Bounding boxes for all images.
      """
      all_boxes = []
      for i in range(0, len(image_paths), batch_size):
          batch_paths = image_paths[i:i + batch_size]
          try:
              # Get bounding boxes for the current batch
              batch_results = self.model(batch_paths)
              for result in batch_results:
                  # Move tensor to CPU before converting to NumPy
                  boxes = result.boxes.xyxy.cpu().numpy()  # Ensure CPU conversion
                  all_boxes.append(boxes)
          except Exception as e:
              print(f"Error processing batch {i // batch_size}: {e}")
      return all_boxes

    @staticmethod
    def pad_bounding_boxes(self, boxes, max_boxes):
      """
      Pad the bounding boxes for a single image to ensure consistent dimensions.
      Args:
          boxes (list): List of bounding boxes for one image.
          max_boxes (int): Maximum number of bounding boxes in the dataset.
      Returns:
          np.ndarray: Padded array of bounding boxes with shape (max_boxes, 4).
      """
      padded = np.zeros((max_boxes, 4))  # Create a zero array for padding
      padded[:len(boxes), :] = boxes    # Fill with actual bounding boxes
      return np.array(padded)

    def predict(self, image_paths):
        """
        Predict bounding boxes for a list of images.
        Args:
            image_paths (List[str]): Paths to input images.
        Returns:
            np.ndarray: Padded bounding boxes for each image.
        """
        # Batch inference using YOLO model
        results = self.model(image_paths, conf=self.conf_threshold, iou=self.iou_threshold)
        all_boxes = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy.size(0) > 0 else np.zeros((0, 4))
            all_boxes.append(boxes)

        # Determine max number of boxes across all images
        max_boxes = max(len(boxes) for boxes in all_boxes)

        # Pad all bounding boxes to the same length
        padded_boxes = [self.pad_bounding_boxes(boxes, max_boxes) for boxes in all_boxes]

        return np.array(padded_boxes)


#player_inference = PlayerInference()
#detections = player_inference.detect("/Users/alyazouzou/Desktop/CV_Football/vids/inf.png", save_output=True)
#print(detections)