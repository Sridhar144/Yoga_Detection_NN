# AI-Powered Yoga Trainer

## Overview
This AI-driven tool provides real-time yoga posture analysis and feedback. It uses pose detection to assess alignment, giving corrective suggestions using text and voice feedback. The tool also generates a comprehensive feedback report.

## Key Features
- **Pose Detection**: Detects and analyzes key yoga poses.
- **Real-Time Feedback**: Provides corrective suggestions based on posture.
- **Text-to-Speech**: Gives voice feedback using `pyttsx3`.
- **AI Feedback**: Generates personalized feedback using GPT-2.
- **Report Generation**: Creates a downloadable feedback report.

## Use Case: Pose Detection & Correction
- Detects yoga poses in real-time.
- Provides feedback on pose alignment and accuracy.

## MODELS USED in the yolo code
# pose detection
[`code`](https://github.com/WongKinYiu/yolov7/tree/pose) [`yolov7-w6-pose.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)


| [**YOLOv7**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) | 640 | **51.4%** | **69.7%** | **55.9%** | 161 *fps* | 2.8 *ms* |


## Technologies Used
- **YOLOv7**: For detecting yoga poses.
- **OpenCV** (`cv2`): Video capture, processing, and feedback overlay.
- **MediaPipe**: Pose detection for analyzing body landmarks.
- **GPT-2**: AI model for generating feedback on posture.
- **pyttsx3**: Text-to-speech engine for voice feedback.
- **MoviePy**: For video file handling.
- **Streamlit**: For the web interface.

## How It Works
1. **Video Upload**: Upload a yoga video for analysis.
2. **Pose Detection**: MediaPipe detects body landmarks in each frame.
3. **Posture Analysis**: Feedback is generated based on key poses and alignment.
4. **Real-Time Feedback**: Suggestions are shown on the video and provided as voice feedback.
5. **Feedback Report**: A report with feedback and scores is generated and available for download.

## Installation

1. **Clone the Repository**:
   ```
   cd yoga-pose-detection
Install Dependencies:



pip install -r requirements.txt
##  Run the Application: Launch the app:



streamlit run app.py

## Code Overview
- Pose Detection: Uses MediaPipe's Pose model to detect key body landmarks.
- Posture Feedback: The angle between key body points (e.g., knees, shoulders) is calculated and compared to ideal values for proper posture.
- AI Feedback: GPT-2 generates feedback based on the analysis of the user's posture.
- Voice Feedback: Text-to-speech engine (pyttsx3) provides spoken feedback on posture alignment.
Example Usage
- Upload a video: Choose a yoga video to analyze.
- Analyze posture: The app will provide visual feedback overlaid on the video.
- Download report: A detailed report with feedback and scores is available for downloa