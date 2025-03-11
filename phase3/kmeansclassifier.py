import cv2
import numpy as np
import torch
from more_itertools import chunked
import pandas as pd
import sys
import os
from sklearn.metrics import silhouette_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mpl_toolkits.mplot3d import Axes3D
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
            #print(f"ℹ️ Extracted {len(crops)} ")

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
        #print(crops)
        #print(f"ℹ️ Processing {len(crops)} crops...")

        if len(crops) == 0:
            print("⚠️ No valid crops to process! Returning empty features.")
            return np.array([])

        valid_crops = [crop for crop in crops if isinstance(crop, np.ndarray) and crop.size > 0 and crop.shape[1] > 0]
        
        if len(valid_crops) == 0:
            print("⚠️ All crops were empty! Returning empty features.")
            return np.array([])

        #print(f"✅ Processing {len(valid_crops)} valid crops...")

        # Convert valid crops to PIL images
        valid_crops = [sv.cv2_to_pillow(crop) for crop in valid_crops]

        batches = chunked(valid_crops, BATCH_SIZE)

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
        Entraîne le modèle KMeans avec les features extraites.

        Args:
            features (np.ndarray): Les features extraites des crops.
        """
        if features.shape[0] == 0:
            print("⚠️ Pas de features valides, impossible d'entraîner le KMeans.")
            return
        
        print("🔄 Entraînement de KMeans...")
        self.kmeans.fit(features)  # Utilise self.kmeans défini dans __init__
        self.trained = True  # Marquer comme entraîné
        print("✅ KMeans entraîné avec succès !")

    def predict_clusters(self, features):
        """
        Prédit les clusters après entraînement du KMeans.

        Args:
            features (np.ndarray): Les features des joueurs.

        Returns:
            np.ndarray: Les labels de cluster (N,)
        """
        if not hasattr(self, 'trained') or not self.trained:
            print("❌ KMeans n'a pas été entraîné ! Exécute `train_kmeans(features)` en premier.")
            return None
        
        return self.kmeans.predict(features)  # Utilise self.kmeans

    
    def projection_umap(self, features, n_components=2):
        """
        Projects the features to a lower-dimensional space using UMAP.

        Args:
            features (np.ndarray): The feature embeddings (N, D).
            n_components (int): Number of dimensions for UMAP (2D or 3D).

        Returns:
            np.ndarray: UMAP projected data.
        """
        if not isinstance(features, np.ndarray):
            raise TypeError(f"Expected features to be a NumPy array, got {type(features)}")

        self.features = features  # Store features
        self.umap_model = umap.UMAP(n_components=n_components)  # ✅ Store UMAP model
        projection = self.umap_model.fit_transform(features)  # Fit once
        self.projection = projection  # Store the projection

        return projection  # Return the transformed features

    
    def plot_projection(self, projection, clusters):
        """
        Plots the UMAP projection in 2D with clusters.

        Args:
            projection (np.ndarray): 2D projected features.
            clusters (np.ndarray): Cluster labels from KMeans.
        """
        if projection.shape[1] != 2:
            raise ValueError("Projection must be 2D for this function. Use n_components=2 in projection_umap().")

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(projection[:, 0], projection[:, 1], c=clusters, cmap="viridis", alpha=0.7)

        if not hasattr(self, 'features'):
            raise AttributeError("❌ `self.features` not found! Run `projection_umap()` first.")

        # ✅ Use the stored UMAP model to transform centroids
        projected_centers = self.umap_model.transform(self.kmeans.cluster_centers_)

        # Plot cluster centers
        plt.scatter(projected_centers[:, 0], projected_centers[:, 1], c='red', marker='x', s=200, label="Centroids")

        plt.colorbar(scatter, label="Cluster Label")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.title("2D UMAP Projection of Player Features")
        plt.legend()
        plt.show()


    def plot_projection_3d(self, projection, clusters):
        """
        Plots the UMAP projection in 3D with clusters.

        Args:
            projection (np.ndarray): 3D projected features.
            clusters (np.ndarray): Cluster labels from KMeans.
        """
        if projection.shape[1] != 3:
            raise ValueError("Projection must be 3D for this function. Use n_components=3 in projection_umap().")

        if not hasattr(self, 'umap_model'):
            raise AttributeError("❌ UMAP model not found! Run `projection_umap()` first.")

        # ✅ Transform centroids using stored UMAP model
        projected_centers = self.umap_model.transform(self.kmeans.cluster_centers_)

        # Compute intra-cluster variance
        variance_intra = np.sum(np.linalg.norm(projection - projected_centers[clusters], axis=1) ** 2)
        print("Variance intra-cluster:", variance_intra)


        # Compute global centroid (mean of projected points)
        global_centroid = np.mean(projection, axis=0)

        # Compute number of samples per cluster
        n_samples_per_cluster = np.bincount(self.kmeans.labels_)

        # Compute inter-cluster variance
        variance_inter = np.sum(n_samples_per_cluster * np.linalg.norm(projected_centers - global_centroid, axis=1) ** 2)
        print("Variance inter-cluster:", variance_inter)

        # Create 3D plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of projected points
        scatter = ax.scatter(projection[:, 0], projection[:, 1], projection[:, 2], 
                            c=clusters, cmap="viridis", alpha=0.7)

        # Plot transformed cluster centers
        ax.scatter(projected_centers[:, 0], projected_centers[:, 1], projected_centers[:, 2], 
                c='red', marker='x', s=200, label="Centroids")

        # Labels and Title
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.set_zlabel("UMAP Dimension 3")
        ax.set_title("3D UMAP Projection of Player Features")
        plt.legend()
        plt.colorbar(scatter, label="Cluster Label")
        plt.show()

    def evaluate(self):
        """
        Evaluate the performance of the KMeans model using various clustering metrics.
        """
        if not hasattr(self, 'kmeans') or not self.kmeans:
            print("❌ KMeans model not trained yet. Train first before evaluation!")
            return
        
        if self.kmeans.n_clusters < 2:
            print("❌ Clustering evaluation requires at least 2 clusters!")
            return

        # Ensure features exist
        if not hasattr(self, 'features'):
            print("❌ Features not found! Run `projection_umap()` first.")
            return

        print(f"📊 Evaluating KMeans with {self.kmeans.n_clusters} clusters")

        # ✅ Transform centroids using stored UMAP model
        projected_centers = self.umap_model.transform(self.kmeans.cluster_centers_)
        # Predict clusters
        predicted_clusters = self.kmeans.predict(self.features)

        # Compute intra-cluster variance
        variance_intra = np.sum(np.linalg.norm(self.projection - projected_centers[predicted_clusters], axis=1) ** 2)
        print("Variance intra-cluster:", variance_intra)

        # Compute global centroid (mean of projected points)
        global_centroid = np.mean(self.projection, axis=0)

        # Compute number of samples per cluster
        n_samples_per_cluster = np.bincount(predicted_clusters)

        # Compute inter-cluster variance
        variance_inter = np.sum(n_samples_per_cluster * np.linalg.norm(projected_centers - global_centroid, axis=1) ** 2)
        print("Variance inter-cluster:", variance_inter)

        ## 2️⃣ **Silhouette Score**
        if self.kmeans.n_clusters > 1:
            silhouette_avg = silhouette_score(self.features, predicted_clusters)
            print(f"🔹 Silhouette Score: {silhouette_avg:.4f} (Higher is better)")
        else:
            print("⚠️ Silhouette Score requires at least 2 clusters.")

        separation_ratio = variance_inter / variance_intra
        print(f"📊 Ratio de Séparation: {separation_ratio:.2f}")

        ## 3️⃣ **Elbow Method (Finding Optimal Clusters)**
        distortions = []
        cluster_range = range(1, 8)  # Testing from 1 to 8 clusters (adjustable)

        for k in cluster_range:
            temp_kmeans = KMeans(n_clusters=k, random_state=42)
            temp_kmeans.fit(self.features)
            distortions.append(temp_kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(cluster_range, distortions, marker="o", linestyle="--")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Distortion (Inertia)")
        plt.title("Elbow Method for Optimal K Selection (1 to 8)")
        plt.grid()
        plt.show()

        print("✅ Evaluation completed.")


#if __name__ == "__main__":

    #video_path = "/Users/alyazouzou/Desktop/CV_Football/vids/good.mov"
    #classifier = kmeansclassifier(video_path, model_path="/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase1/runs/detect/train/weights/best.pt") 
 
    #crops = classifier.get_crops_from_frames(stride=150, player_id=2)

    # Extract CLIP features
    #features = classifier.get_features(crops)

    # Train KMeans on features
    #classifier.train_kmeans(features)

    #projection = classifier.projection_umap(features)  # Returns only one value

    # Then use predicted clusters separately
    #clusters = classifier.predict_clusters(features)

    # Now plot using both values
    #classifier.plot_projection(projection, clusters)

    #classifier.evaluate()
    
    #if crops:
        #classifier.plot_crops(crops)
