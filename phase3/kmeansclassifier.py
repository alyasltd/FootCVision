import cv2
import numpy as np
import torch
from more_itertools import chunked
import pandas as pd
import sys
import os

# Get the absolute path of the parent directory (FootCVision)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase1.inference import PlayerInference
import supervision as sv 
from tqdm.auto import tqdm  
from transformers import CLIPProcessor, CLIPModel
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class kmeansclassifier:
    def __init__(self, video_path, n_clusters=2, model_path="/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase1/runs/detect/train/weights/best.pt", conf_threshold=0.8, iou_threshold=0.8):
        """
        Initialize the class for extracting player crops from a video using YOLO detection. TO RE DO 

        Args:
            video_path (str): Path to the input video.
            model_path (str): Path to the trained YOLO model weights.
            conf_threshold (float): Confidence threshold for detections.
            iou_threshold (float): IoU threshold for non-max suppression.
        """
        self.video_path = video_path
        self.detector = PlayerInference(model_path, conf_threshold, iou_threshold)  # Use PlayerInference
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Ensures we always get consistent results


    def get_crops_from_frames(self, stride=30, player_id=2):
        """
        Extract player crops from video frames using PlayerInference.

        Args:
            stride (int): Number of frames to skip between processed frames.
            player_id (int): Class ID for players.

        Returns:
            List of cropped player images.
        """
        frame_generator = sv.get_video_frames_generator(
            source_path=self.video_path, stride=stride)

        crops = []
        for frame in tqdm(frame_generator, desc="Extracting player crops"):
            detections = self.detector.inference(frame)  # Use PlayerInference for detection

            # Apply Non-Maximum Suppression (NMS) and filter only player detections
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            players_detections = detections[detections.class_id == player_id]
            
            # Extract player crops
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            crops += players_crops  # Add crops to the list

        return crops


    def plot_crops(self, crops):
        """
        Display extracted player crops in a grid.

        :param crops: List of cropped player images.
        """
        if not crops:
            print("No crops to display.")
            return

        # Display all crops (up to 100) in a grid
        sv.plot_images_grid(crops[:100], grid_size=(10, 10))

    def get_features(self, crops):
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        BATCH_SIZE = 32
        print(f"ℹ️ Processing {len(crops)} crops...")

        if len(crops) == 0:
            print("⚠️ No valid crops to process! Returning empty features.")
            return np.array([])  # Return an empty array to avoid crashes
        
        # Convert valid crops to PIL images
        crops = [sv.cv2_to_pillow(crop) for crop in crops]


        batches = chunked(crops, BATCH_SIZE)

        data = []
        with torch.no_grad():  # Disable gradients for faster inference
            for batch in tqdm(batches, desc="Extracting embeddings"):
                inputs = processor(images=batch, return_tensors="pt").to(device)
                outputs = model.vision_model(**inputs)
                embeddings = outputs.pooler_output.cpu().numpy()  # Extract image embeddings only
                data.append(embeddings)

        # Concatenate all embeddings
        data = np.concatenate(data) if data else np.array([])
        return data

    
    def train_kmeans(self, features):
        """
        Trains the KMeans model on extracted features and saves centroids.

        Args:
            features (np.ndarray): The feature embeddings (N, 768).
        """
        if features.shape[0] == 0:
            print("No features to train KMeans.")
            return
        self.kmeans.fit(features)
        self.trained = True  # Mark as trained

    def predict_clusters(self, features):
        """
        Predicts the cluster labels (team 0 or 1) based on extracted features.

        Args:
            features (np.ndarray): The feature embeddings (N, 768).

        Returns:
            np.ndarray: Cluster labels (N,) as 0 or 1.
        """
        return self.kmeans.predict(features)  # Outputs (N,)

    def projection_umap(self, features):
        """
        Projects the features to a 2D space using UMAP for visualization.

        Args:
            features (np.ndarray): The feature embeddings (N, 768).

        Returns:
            np.ndarray: UMAP projected 2D data.
        """
        return umap.UMAP(n_components=2).fit_transform(features)
   

    def plot_projection(self, projection, clusters):
        """
        Plots the UMAP projection with clusters.

        :param projection: 2D array of projected features (output from UMAP).
        :param clusters: Cluster labels from KMeans.
        """
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(projection[:, 0], projection[:, 1], c=clusters, cmap="viridis", alpha=0.7)

        # Add cluster centers
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(projection)
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label="Centroids")

        plt.colorbar(scatter, label="Cluster Label")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.title("UMAP Projection of Player Features")
        plt.legend()
        plt.show()


    #evaluation model performance - clustering variance intra and inter
    #evaluation model performance - silhouette score
    #evaluation model performance - elbow method


#if __name__ == "__main__":

    #video_path = "/Users/alyazouzou/Desktop/CV_Football/vids/good.mov"
    #classifier = kmeansclassifier(video_path, model_path="/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase1/runs/detect/train/weights/best.pt") 
 
    #crops = classifier.get_crops_from_frames(stride=150, player_id=2)

    # Extract CLIP features
    #features = classifier.get_features(crops)

    # Compute UMAP projection and clusters
    #projection, clusters = classifier.projection_umap(features)

    # Plot UMAP projection with clusters
    #classifier.plot_projection(projection, clusters)

    # Extract CLIP features
    #features = classifier.get_features(crops)

    # Train KMeans on features
    #classifier.train_kmeans(features)

    # Predict cluster labels
    #clusters = classifier.predict_clusters(features)

    # Compute UMAP projection
    #projection = classifier.projection_umap(features)

    # Plot UMAP projection with clusters
    #classifier.plot_projection(projection, clusters)

    
    #if crops:
        #classifier.plot_crops(crops)
