#the referee is not detected in the video
import cv2
import numpy as np
import matplotlib.pyplot as plt

class hsvclassifier : 
    def __init__(self, video_path, df_tracking): 
        self.video_path = video_path
        self.df_tracking = df_tracking
        self.class_labels = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
        self.color_ranges = {
           "liverpool": [(0,50,20),(5,255,255)],
           "man_united": self.get_hsv_range_from_image('/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase3/manunited.png'),
        }
    
    def collect_and_classify(self, video_path, df_tracking): 
        #collect bbox from df_tracking
        #crop each image for each frame 
        #extract main color from crop mask
        #classify the color
        #return the classified df_tracking
        pass 
    
    #crop image function 
    # extract main color from crop mask 



    # image à faire - celle du mask celle où on a le maillot de l'équipe et celle où on a ploter dans l'espace hsv le maillot     

    def extract_colors(self, bounded_box, frame): 
        x, y, w, h = bounded_box
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate the histogram of the HSV ROI
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Find the peak in the histogram
        dominant_color = np.unravel_index(np.argmax(hist), hist.shape)
        
        return dominant_color


    def plot(df_classified): 
        pass


# Example usage
if __name__ == "__main__":
    # Get HSV ranges for Liverpool and Manchester United
    classifier = hsvclassifier(video_path='/Users/alyazouzou/Desktop/CV_Football/vids/good.mov', df_tracking='/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase2/test.csv')
    
    range_liverpool = classifier.get_hsv_range_from_image('/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase3/liverpool.png')
    range_manunited = classifier.get_hsv_range_from_image('/Users/alyazouzou/Desktop/CV_Football/FootCVision/phase3/manunited.png')
    
    # Plot the HSV ranges
    classifier.plot_hsv_range(range_liverpool)
    classifier.plot_hsv_range(range_manunited)