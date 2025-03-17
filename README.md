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
  <img src="./utils/img/cp_close.png" alt="CP Image" width="50%" />
</div>
 
---

### **Phase 2: Ball Tracking and Player Statistics** 🎥⚽

1. **Ball & Player Tracking** ⚽
The first video is ball & player tracking with HSV Classifier and the second is from the Kmeans classifier.
<div align="center">
<img src="./utils/vid/output_hsv.gif" width="600" /> 
<img src="./utils/vid/output_kmeans.gif" width="600" />
</div>

2. **Player Statistics** 📈

---

### **Phase 3: Two Approaches for Team Differentiation** 📊
   - **HSV Classifier**:
     - Extract **HSV colors** from bounding box regions of detected players.
     - Cluster players into teams based on dominant uniform colors.
<div align="center">
  <img src="./utils/img/hsv_track_ex.png" alt="CP Image" width="60%" />
</div>

   - **K-Means Clustering for Team Analysis**:
     - Use player positions (bounding box coordinates) and CLIP features to cluster players into two teams.
<div align="center">
  <img src="./utils/img/kmeans_track_ex.png" alt="CP Image" width="60%" />
</div>       
---

### **Phase 4: Offside Detection** 🚩

1. **Rule-Based Offside Detection**
   - Utilize player positions and ball location to detect offside scenarios.
   - Apply rule-based AI to automate offside decision-making.

---

## **Installation and Usage** 🚀

### **Clone the Project**
```bash
git clone https://github.com/alyasltd/FootCVision.git
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```
