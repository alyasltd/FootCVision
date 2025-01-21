import os
import pickle
import numpy as np
from tqdm import tqdm
from itertools import compress
from ultralytics import YOLO
from deel.puncc.api.utils import hungarian_assignment

class YOLOAPIWrapper:
    def __init__(self, model_path, file_path="calibration_results.pickle", conf_threshold=0.5, iou_threshold=0.5):
        """
        Initialize the YOLO model and file path for saving/loading results.
        Args:
            model_path (str): Path to the trained YOLO model weights.
            file_path (str): Path to save/load calibration data.
            conf_threshold (float): Confidence threshold for detections.
            iou_threshold (float): IoU threshold for non-max suppression.
        """
        self.model = YOLO(model_path)
        self.file_path = file_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def predict_from_image(self, image_path): 
        """
        Predict bounding boxes for a single image.
        Args:
            image_path (str): Path to the input image.
        Returns:
            np.ndarray: Predicted bounding boxes in (x1, y1, x2, y2) format.
        """
        results = self.model(image_path, conf=self.conf_threshold, iou=self.iou_threshold)
        boxes = []
        
        for result in results:
            boxes.extend(result.boxes.xyxy.cpu().numpy())  # Ensure numpy format
        
        # Ensure the returned array has shape (0, 4) if no detections
        if len(boxes) == 0:
            return np.zeros((0, 4))
        
        return np.array(boxes)

    def predict_and_match(self, image_path, y_trues_per_image, min_iou=0.1):
        """
        Predict bounding boxes and match them with true bounding boxes using the Hungarian algorithm.
        Args:
            image_path (str): Path to the input image.
            y_trues_per_image (np.ndarray): True bounding boxes for the image.
            min_iou (float): Minimum IoU for valid matches.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Matched predicted and true bounding boxes.
        """

        y_preds_per_image = self.predict_from_image(image_path)
        if y_preds_per_image is None or y_preds_per_image.shape[0] == 0:
            return np.zeros((0, 4)), np.zeros((0, 4)), np.array([], dtype=bool)
        
        y_preds_i, y_trues_i, indices_i = hungarian_assignment(
            np.array(y_preds_per_image), np.array(y_trues_per_image), min_iou=min_iou
        )
        return y_preds_i, y_trues_i, indices_i

    def load_results(self):
        """
        Load previously saved results from a file.
        Returns:
            Tuple: y_preds, y_trues, images, and labels.
        """
        if os.path.exists(self.file_path):
            with open(self.file_path, "rb") as file:
                results_dict = pickle.load(file)
                return (
                    results_dict["y_preds"],
                    results_dict["y_trues"],
                    results_dict["images"],
                    results_dict["labels"],
                )
        else:
            raise FileNotFoundError(f"No results file found at {self.file_path}.")

    def save_results(self, y_preds, y_trues, images, labels):
        """
        Save results to a file.
        Args:
            y_preds (list): Predicted bounding boxes.
            y_trues (list): True bounding boxes.
            images (list): Image paths.
            labels (list): Associated labels or metadata.
        """
        with open(self.file_path, "wb") as file:
            pickle.dump(
                {"y_preds": y_preds, "y_trues": y_trues, "images": images, "labels": labels}, file
            )

    def query(self, image_paths, y_trues, labels, min_iou=0.4, n_instances=None):
        """
        Predict bounding boxes for a batch of images and match them to ground truth using Hungarian assignment.
        Args:
            image_paths (List[str]): List of image paths.
            y_trues (List[np.ndarray]): List of true bounding boxes for each image.
            labels (List[List[int]]): List of true labels for each bounding box.
            min_iou (float): Minimum IoU for valid matches.
            n_instances (int): Maximum number of images to process. If None, process all images.
        Returns:
            Tuple[np.ndarray, np.ndarray, list, np.ndarray]: Predictions, ground truths, image paths, and labels.
        """
        y_preds, matched_trues, images, classes = [], [], [], []

        # Check if results already exist
        if os.path.exists(self.file_path):
            return self.load_results()

        # Iterate over the dataset
        for counter, (image_path, y_true, label) in enumerate(
            tqdm(zip(image_paths, y_trues, labels), total=len(image_paths))
        ):
            # Predict and match bounding boxes
            y_preds_i, y_trues_i, indices_i = self.predict_and_match(image_path, y_true, min_iou=min_iou)
            y_preds.append(y_preds_i)
            matched_trues.append(y_trues_i)
            images.append(image_path)

            # Get matched classes
            #print(label)
            classes.append(list(compress(label, indices_i)))
            #print("cc")
            #print(classes)
            # Stop if n_instances is reached
            if n_instances is not None and counter + 1 >= n_instances:
                break

        # Concatenate results
        y_preds = np.concatenate(y_preds, axis=0) #if y_preds else np.array([])
        matched_trues = np.concatenate(matched_trues, axis=0) #if matched_trues else np.array([])
        classes = np.concatenate(classes, axis=0) #if classes else np.array([])

        # Save results
        self.save_results(y_preds, matched_trues, images, classes)

        return y_preds, matched_trues, images, classes

