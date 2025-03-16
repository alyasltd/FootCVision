# ⚽️ FOOTCVISION : Computer Vision Applied to Football

## **Roadmap** 🛣️

### **Phase 1: Player Detection and Conformal Prediction** 🏃‍♂️⚽️

1. **YOLOv11 Fine-Tuning for Player Detection** 🎯
<!-- Center the image using HTML -->
<div align="center">
  <img src="./utils/img/infer.png" alt="Inference Image" width="60%" />
</div>

2. **Conformal Object Detection with puncc library** 📏
<div align="center">
  <img src="./utils/img/cp_close.png" alt="CP Image" width="60%" />
</div>
 
---

### **Phase 2: Ball Tracking and Player Statistics** 🎥⚽

1. **Ball Tracking** ⚽
  <!-- Videos side by side using HTML -->
<div align="center">
  <video width="400" controls>
    <source src="./utils/vid/gif_hsv.mp4" type="video/mp4">
  </video>
  <video width="400" controls>
    <source src="./utils/vid/gif_kmeans.mp4" type="video/mp4">
  </video>
</div>

2. **Player Statistics** 📈
   - Extract metrics for each team, such as:
     - Distance covered.
     - Ball possession time.
     - Speed and acceleration.

---

### **Phase 3: Two Approaches for Team Differentiation** 📊
   - **HSV Classifier**:
     - Extract **color histograms** from bounding box regions of detected players.
     - Cluster players into teams based on dominant uniform colors.
     - **Objective**: Provide a visually interpretable method for distinguishing teams based on color features.

   - **K-Means Clustering for Team Analysis**:
     - Use player positions (bounding box coordinates) and spatial distributions to cluster players into two teams.
     - **Objective**: Offer a complementary approach to visualizing teams, focusing on spatial and positional analysis.
---

### **Phase 4: Offside Detection** 🚩

1. **Rule-Based Offside Detection**
   - Utilize player positions and ball location to detect offside scenarios.
   - Apply rule-based AI to automate offside decision-making.

2. **Real-Time Offside Detection**
   - Develop a pipeline for real-time offside detection during live matches.


---

## **Tech Stack** 🛠️

- **Object Detection**: YOLOv11 (PyTorch).
- **Color Analysis**: OpenCV for histogram extraction.
- **Clustering**: K-Means for spatial team analysis.
- **Tracking**: DeepSORT or ByteTrack for ball tracking.
- **Uncertainty Quantification**: Punch library for conformal prediction.

---

## **Installation and Usage** 🚀

### **Clone the Project**
```bash
git clone https://github.com/alyasltd/FootCVision2.git
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```
