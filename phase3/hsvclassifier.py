import cv2
import numpy as np

class HSVClassifier:
    def __init__(self):
        """
        Initialise le classificateur HSV avec des plages de couleurs prédéfinies pour les équipes.
        """
        self.color_ranges = {
            0: [(17, 0, 138), (122, 113, 255)],  # Team 0 (ex: Manchester United)
            1: [(18, 0, 136), (129, 116, 255)]   # Team 1 (ex: Liverpool)
        }

    def get_hsv_img(self, img: np.ndarray) -> np.ndarray:
        """
        Convertit une image en espace de couleur HSV.
        """
        return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    def apply_filter(self, img: np.ndarray, color_range: tuple) -> np.ndarray:
        """
        Applique un masque de couleur HSV sur l'image.
        """
        lower_hsv, upper_hsv = color_range
        img_hsv = self.get_hsv_img(img)
        mask = cv2.inRange(img_hsv, np.array(lower_hsv), np.array(upper_hsv))
        return mask

    def classify_player(self, img: np.ndarray) -> int:
        """
        Classifie le joueur en fonction de la couleur dominante de son maillot.

        Retourne :
        - 0 si la couleur dominante correspond à l'équipe 0.
        - 1 si la couleur dominante correspond à l'équipe 1.
        """
        max_pixels = 0
        predicted_team = 0  # Valeur par défaut

        for team_id, color_range in self.color_ranges.items():
            mask = self.apply_filter(img, color_range)
            non_black_pixels = cv2.countNonZero(mask)

            if non_black_pixels > max_pixels:
                max_pixels = non_black_pixels
                predicted_team = team_id  # On garde la meilleure correspondance

        return predicted_team