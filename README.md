# ⚽️ Computer Vision Applied to Football

## **Roadmap** 🛣️

### **Phase 1: Player Detection and Initial Analysis** 🏃‍♂️⚽️

1. **YOLOv11 Fine-Tuning for Player Detection** 🎯
   - Adapt YOLOv11 for detecting football players on the field.
   - **Objective**: Ensure robust detection of players, even under challenging scenarios like crowded scenes or varying lighting conditions.

2. **Conformal Object Detection with puncc library** 📏
 
---

### **Phase 2: Ball Tracking and Player Statistics** 🎥⚽

1. **Ball Tracking** ⚽
   - Implement tracking algorithms (e.g., **DeepSORT** or **ByteTrack**) to follow the ball's movement across frames.
   - Extract trajectories to analyze key events like passes, shots, and goals.

2. **Player Statistics** 📈
   - Extract metrics for each player, such as:
     - Distance covered.
     - Ball possession time.
     - Speed and acceleration.

---

### **Phase 3: **Two Approaches for Team Differentiation** 📊
   - **Color Histograms for Team Identification**:
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
