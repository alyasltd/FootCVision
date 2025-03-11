import cv2
import numpy as np
from typing import List, Tuple
from tqdm.auto import tqdm
import supervision as sv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from phase1.inference import PlayerInference  # Détecteur YOLO

class HSVClassifier:
    def __init__(self, video_path: str, model_path: str, conf_threshold=0.8, iou_threshold=0.8):
        """
        Initialise le classificateur HSV avec un détecteur de joueurs.

        Args:
            video_path (str): Chemin de la vidéo.
            model_path (str): Chemin du modèle YOLO.
            conf_threshold (float): Seuil de confiance pour les détections.
            iou_threshold (float): Seuil IoU pour la suppression des doublons.
        """
        self.video_path = video_path
        self.detector = PlayerInference(model_path, conf_threshold, iou_threshold) 
        self.color_ranges = {
            "manchester_united": [(0, 70, 50), (10, 255, 255)],  # Équipe 0 (ex: Manchester United)
            "liverpool": [(0,0,168), (172,111,255)]   # Équipe 1 (ex: Liverpool)
        }

    def get_crops_from_frames(self, stride=30, player_id=2) -> List[np.ndarray]:
        """
        Extrait les crops des joueurs à partir des frames d'une vidéo.

        Args:
            stride (int): Nombre de frames à sauter entre deux traitements.
            player_id (int): ID de classe des joueurs.

        Returns:
            List[np.ndarray]: Liste des crops de joueurs extraits.
        """
        frame_generator = sv.get_video_frames_generator(
            source_path=self.video_path, stride=stride)

        crops = []
        for frame in tqdm(frame_generator, desc="Extracting player crops"):
            detections = self.detector.inference(frame)  # Détection des joueurs

            # Appliquer le NMS et filtrer uniquement les joueurs
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            players_detections = detections[detections.class_id == player_id]

            # Extraire les crops des joueurs détectés
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            crops.extend(players_crops)  # Ajouter les crops à la liste

        return crops

    def crop_img_for_jersey(self, img: np.ndarray) -> np.ndarray:
        """
        Coupe l'image pour se concentrer uniquement sur la partie du maillot.

        Args:
            img (np.ndarray): Image du joueur.

        Returns:
            np.ndarray: Image du maillot.
        """
        height, width, _ = img.shape

        # Découpe pour garder le haut du corps (maillot)
        y_start = int(height * 0.15)
        y_end = int(height * 0.50)
        x_start = int(width * 0.15)
        x_end = int(width * 0.85)

        return img[y_start:y_end, x_start:x_end]

    def convert_to_hsv(self, img: np.ndarray) -> np.ndarray:
        """
        Convertit une image en espace de couleur HSV.

        Args:
            img (np.ndarray): Image en format BGR.

        Returns:
            np.ndarray: Image en format HSV.
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def apply_hsv_filter(self, hsv_img: np.ndarray, lower: Tuple[int, int, int], upper: Tuple[int, int, int]) -> np.ndarray:
        """
        Applique un masque HSV pour extraire une couleur spécifique.

        Args:
            hsv_img (np.ndarray): Image en format HSV.
            lower (tuple): Seuil inférieur de couleur HSV.
            upper (tuple): Seuil supérieur de couleur HSV.

        Returns:
            np.ndarray: Masque binaire de la couleur détectée.
        """
        return cv2.inRange(hsv_img, np.array(lower), np.array(upper))

    def add_median_blur(self, img: np.ndarray) -> np.ndarray:
        """
        Applique un flou médian pour lisser les couleurs et réduire le bruit.

        Args:
            img (np.ndarray): Image à flouter.

        Returns:
            np.ndarray: Image après flou médian.
        """
        return cv2.medianBlur(img, 5)

    def extract_colored_regions(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Extrait les zones correspondant aux couleurs des équipes.

        Args:
            img (np.ndarray): Image d'entrée.

        Returns:
            List[np.ndarray]: Liste des images segmentées par couleur.
        """
        hsv_img = self.convert_to_hsv(img)
        color_masks = []

        for team_id, (lower, upper) in self.color_ranges.items():
            mask = self.apply_hsv_filter(hsv_img, lower, upper)
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            color_masks.append((team_id, masked_img))

        return color_masks

    def count_non_black_pixels(self, img: np.ndarray) -> int:
        """
        Compte les pixels non noirs dans une image.

        Args:
            img (np.ndarray): Image filtrée.

        Returns:
            int: Nombre de pixels non noirs.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.countNonZero(gray)

    def crop_filter_and_blur_img(self, img: np.ndarray) -> np.ndarray:
        """
        Découpe l'image pour récupérer le maillot, applique le filtre couleur et ajoute un flou.

        Args:
            img (np.ndarray): Image d'entrée.

        Returns:
            np.ndarray: Image transformée.
        """
        transformed_img = self.crop_img_for_jersey(img)
        transformed_img = self.convert_to_hsv(transformed_img)
        transformed_img = self.add_median_blur(transformed_img)
        return transformed_img

    def predict_team(self, img: np.ndarray) -> int:
        """
        Détermine l'équipe du joueur en analysant les couleurs dominantes.

        Args:
            img (np.ndarray): Image du joueur.

        Returns:
            int: ID de l'équipe (0 ou 1).
        """
        img = self.crop_filter_and_blur_img(img)  # Extraire la zone du maillot et flouter
        color_masks = self.extract_colored_regions(img)

        max_pixels = 0
        best_team = -1  # Par défaut : inconnu

        for team_id, masked_img in color_masks:
            pixels = self.count_non_black_pixels(masked_img)
            if pixels > max_pixels:
                max_pixels = pixels
                best_team = team_id

        return best_team

    def predict(self, stride=30) -> List[Tuple[np.ndarray, int]]:
        """
        Applique tout le pipeline de détection et classification.

        Args:
            stride (int): Nombre de frames à sauter pour optimiser le traitement.

        Returns:
            List[Tuple[np.ndarray, int]]: Liste des crops et leur équipe prédite.
        """
        crops = self.get_crops_from_frames(stride=stride)
        results = [(crop, self.predict_team(crop)) for crop in crops]
        return results

    def display_results(self, results: List[Tuple[np.ndarray, int]]):
        """
        Affiche les résultats des classifications avec OpenCV.

        Args:
            results (List[Tuple[np.ndarray, int]]): Liste des crops et de leurs prédictions.
        """
        for crop, team_id in results:
            label = f"Team {team_id}" if team_id != -1 else "Unknown"
            cv2.imshow(label, crop)
            cv2.waitKey(500)  # Pause pour voir chaque image
        cv2.destroyAllWindows()