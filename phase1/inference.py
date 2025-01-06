from ultralytics import YOLO

class PlayerInference:
    def __init__(self, model_path="/Users/alyazouzou/Desktop/CV_Football/FootCVision2/phase1/runs/detect/train/weights/best.pt", conf_threshold=0.5, iou_threshold=0.5):
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

# Example Usage:
player_inference = PlayerInference()
detections = player_inference.detect("/Users/alyazouzou/Desktop/CV_Football/vids/inf.png", save_output=True)
print(detections)
