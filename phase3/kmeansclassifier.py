import cv2
import numpy as np
import torch
from more_itertools import chunked
import pandas as pd
import supervision as sv 
from tqdm.auto import tqdm  
from transformers import CLIPProcessor, CLIPModel
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class kmeansclassifier:
    def __init__(self, video_path, df_tracking):
        self.video_path = video_path
        self.df_tracking = df_tracking

    def get_crops_from_frames(self, target_frames):
        """
        Extract crops from multiple specified frames using tracking data.
        
        :param target_frames: List of frame numbers from which to extract crops.
        :return: Dictionary {frame_number: list_of_crops} containing cropped player images.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return {}

        all_crops = {}

        # Get video frame dimensions
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read video.")
            cap.release()
            return {}

        for frame_idx in tqdm(target_frames, desc="Extracting crops"):
            frame_data = self.df_tracking[self.df_tracking["frame"] == frame_idx]
            if frame_data.empty:
                print(f"No tracking data found for frame {frame_idx}")
                continue

            # Set video to the target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame {frame_idx}")
                continue

            crops = []
            for _, row in frame_data.iterrows():
                if row["class"] == "player":  # Only consider player crops
                    x_center = row["x"]
                    y_center = row["y"]
                    box_width = row["w"]
                    box_height = row["h"]

                    # Calculate top-left and bottom-right coordinates of the bounding box
                    x_min = int(x_center - box_width / 2)
                    y_min = int(y_center - box_height / 2)
                    x_max = int(x_center + box_width / 2)
                    y_max = int(y_center + box_height / 2)

                    # Ensure crop dimensions are valid
                    if x_max > x_min and y_max > y_min:
                        crop = frame[y_min:y_max, x_min:x_max]
                        if crop.size > 0:  # Ensure crop is not empty
                            crops.append(crop)

            all_crops[frame_idx] = crops  # Store crops for the frame

        cap.release()
        return all_crops


    def plot_crops(self, crops_dict):
        """
        Display all crops from multiple frames in a grid.
        
        :param crops_dict: Dictionary {frame_number: list_of_crops}
        """
        all_crops = []  # List to store all crops

        # Extract crops from all frames
        for frame_idx, crops in crops_dict.items():
            all_crops.extend(crops)  # Add crops from each frame to the list

        if not all_crops:
            print("No crops to display.")
            return

        # Display all crops (up to 100) in a grid
        sv.plot_images_grid(all_crops[:100], grid_size=(10, 10))

    def get_features(self, crops):
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        BATCH_SIZE = 32

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

    def projection_umap(self, features):
        projection = umap.UMAP(n_components=2).fit_transform(features)
        clusters = KMeans(n_clusters=3).fit_predict(projection)
        return projection, clusters
        

    def plot_projection(self, projection, clusters):
        """
        Plots the UMAP projection with clusters.

        :param projection: 2D array of projected features (output from UMAP).
        :param clusters: Cluster labels from KMeans.
        """
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(projection[:, 0], projection[:, 1], c=clusters, cmap="viridis", alpha=0.7)

        # Add cluster centers
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(projection)
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label="Centroids")

        plt.colorbar(scatter, label="Cluster Label")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.title("UMAP Projection of Player Features")
        plt.legend()
        plt.show()


if __name__ == "__main__":

    video_path = "/Users/alyazouzou/Desktop/CV_Football/vids/good.mov"
    df_tracking = pd.read_csv("/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase2/test.csv")
    classifier = kmeansclassifier(video_path, df_tracking )
    selected_frames = [ 5, 10, 20, 18, 30, 64, 36, 89]  
    crops_dict = classifier.get_crops_from_frames(selected_frames)

    #faire une montage  pour avoir les deuc gardiens et avoir 4 clusters 
    # Flatten all crops into a single list
    all_crops = [crop for crops in crops_dict.values() for crop in crops]

    # Extract CLIP features
    features = classifier.get_features(all_crops)

    # Extract features
    features = classifier.get_features(all_crops)

    # Compute UMAP projection and clusters
    projection, clusters = classifier.projection_umap(features)

    # Plot UMAP projection with clusters
    classifier.plot_projection(projection, clusters)
    
    if crops_dict:
        classifier.plot_crops(crops_dict) 
