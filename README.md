# Robust Deepfake Detection using a Hybrid ResNeXt-LSTM Network

This project implements the high-performance deepfake detection model described in the research paper, *"Robust Deepfake Detection Using Resnet and LSTM Networks: A Hybrid Approach for Enhanced Media Integrity."* It uses a hybrid deep learning architecture that combines a **ResNeXt** model for spatial feature extraction and a **Long Short-Term Memory (LSTM)** network for temporal analysis.

This dual approach allows the model to detect not only frame-by-frame visual artifacts but also inconsistencies in motion and transitions over time, leading to state-of-the-art detection accuracy.

## 🚀 Key Features

-   **Hybrid Architecture**: Leverages a ResNeXt CNN to identify spatial anomalies (e.g., texture, lighting) and an LSTM network to analyze temporal inconsistencies between frames.
-   **High Accuracy**: Achieved **96.3% accuracy** on the test dataset, outperforming many standard detection models.
-   **Research-Backed**: The methodology and results are based on a formal research study, providing a strong foundation for the model's effectiveness.
-   **End-to-End Pipeline**: Includes a complete data preprocessing pipeline for face detection, cropping, and normalization.

## 🛠️ Technology Stack

-   **Backend**: Python
-   **Environment**: Jupyter Notebook / Python Scripts
-   **Deep Learning Frameworks**: TensorFlow, Keras
-   **Models**: ResNeXt, LSTM
-   **Libraries**: OpenCV (`cv2`), NumPy, Matplotlib

## 📋 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You need to have Python and `pip` installed. It is highly recommended to use a virtual environment.

```sh
python --version
pip --version
```

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/rishik0821/Deepfake-Detection-.git](https://github.com/rishik0821/Deepfake-Detection-.git)
    ```

2.  **Navigate to the project directory:**
    ```sh
    cd Deepfake-Detection-
    ```

3.  **Create and activate a virtual environment (Recommended):**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install the required packages:**
    ```sh
    pip install notebook tensorflow keras opencv-python numpy matplotlib
    ```

5.  **Download the Dataset**:
    The model was trained on the "Trim Dataset" as mentioned in the paper, but can be adapted for other public datasets like FaceForensics++ or Celeb-DF. Place your dataset in a structured way (e.g., separate folders for `real` and `fake` videos).

## 📖 Model Workflow

The model operates in a two-stage process for comprehensive analysis:

1.  **Spatial Analysis (ResNeXt)**:
    -   Videos are first split into individual frames.
    -   A face detection model identifies and crops the region of interest (ROI) in each frame.
    -   The pre-trained ResNeXt model processes each frame to extract spatial feature vectors, identifying subtle manipulation artifacts like edge inconsistencies and unnatural textures.

2.  **Temporal Analysis (LSTM)**:
    -   The sequence of feature vectors extracted by ResNeXt is fed into the LSTM network.
    -   The LSTM analyzes the sequence to detect temporal anomalies, such as unnatural facial movements or inconsistent transitions between frames.
    -   A final classification layer determines if the video is `real` or `fake`.

## 📈 Performance

The hybrid model demonstrates superior performance compared to using either a spatial or temporal model alone.

| Model                 | Accuracy | Precision | Recall  | F1-Score |
| --------------------- | -------- | --------- | ------- | -------- |
| ResNeXt Only          | 93.0%    | 0.88      | 0.87    | 0.875    |
| LSTM Only             | 52.0%    | 0.50      | 0.49    | 0.487    |
| **Hybrid (Ours)** | **96.3%**| **0.958** | **0.967**| **0.962**|

These results highlight the effectiveness of combining both spatial and temporal analysis for robust deepfake detection.

---
*This README is based on the research paper by Sri Charan, Sai Rishik, Samhith, et al. and was generated with assistance from an AI.*

