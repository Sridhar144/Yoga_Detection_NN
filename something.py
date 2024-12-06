import cv2
import base64
import tempfile
import streamlit as st
import numpy as np
from moviepy.editor import VideoFileClip
import mediapipe as mp
import pyttsx3
import time
import os
import torch
from transformers import pipeline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Text-to-speech engine
engine = pyttsx3.init()

# Load GPT-2 for generating feedback
feedback_model = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)

# Function to analyze the posture and provide feedback
def analyze_posture(landmarks, frame_time):
    feedback = []
    scores = []
    timestamps = []

    # Analyze left knee posture
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    if knee_angle < 160:
        feedback_text = f"At {frame_time}s: Straighten your left knee."
        feedback.append(feedback_text)
        scores.append(-1)
        timestamps.append(frame_time)
    elif knee_angle > 180:
        feedback_text = f"At {frame_time}s: Bend your left knee slightly."
        feedback.append(feedback_text)
        scores.append(-1)
        timestamps.append(frame_time)
    else:
        feedback_text = f"At {frame_time}s: Good alignment for the left knee!"
        feedback.append(feedback_text)
        scores.append(10)

    # Analyze back posture
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    mid_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

    back_angle = calculate_angle(left_shoulder, mid_hip, right_shoulder)
    if abs(back_angle - 180) > 10:
        feedback_text = f"At {frame_time}s: Keep your back straighter for better alignment."
        feedback.append(feedback_text)
        scores.append(-10)
        timestamps.append(frame_time)
    else:
        feedback_text = f"At {frame_time}s: Your back posture looks great!"
        feedback.append(feedback_text)
        scores.append(10)

    # Generate AI feedback based on the analysis
    ai_feedback = generate_ai_feedback(feedback)
    feedback.extend(ai_feedback)
    return feedback, sum(scores), timestamps

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    import math
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# Function to generate AI feedback using GPT-2
def generate_ai_feedback(feedback_list):
    prompt = "Provide feedback on the following yoga posture analysis:\n" + "\n".join(feedback_list) + "\nFeedback:"
    
    # Generate feedback with truncation and padding handling
    generated_feedback = feedback_model(prompt, 
                                       max_length=150, 
                                       num_return_sequences=1,
                                       truncation=True, 
                                       pad_token_id=50256)
    
    return [feedback['generated_text'].strip() for feedback in generated_feedback]


# Function to draw landmarks and connections on the video frame
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )

# Function to give voice feedback
def give_voice_feedback(feedback_text):
    engine.say(feedback_text)
    engine.runAndWait()

# Function to save feedback as a report
def save_feedback_report(feedback, score, timestamps, output_file="feedback_report.txt"):
    with open(output_file, "w") as f:
        f.write("Yoga Posture Feedback Report\n")
        f.write("="*30 + "\n\n")
        for line, timestamp in zip(feedback, timestamps):
            f.write(f"{timestamp}s: {line}\n")
        f.write(f"\nTotal Score: {score}\n")
    return output_file

def main():
    st.title("Enhanced Yoga AI Trainer with Real-Time Feedback")

    # Upload video file
    uploaded_file = st.file_uploader("Choose a yoga video file", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        video_path = "uploaded_video.mp4"
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Display the uploaded video
        st.video(uploaded_file)

        if st.button('Analyze Yoga Posture'):
            st.write("Analyzing... Please wait.")

            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(
                "output.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (frame_width, frame_height)
            )

            total_score = 0
            feedback_list = []
            timestamps_list=[]
            frame_rate = 3  # Process every 5th frame
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % frame_rate != 0:
                    continue
                frame_time = frame_count // frame_rate
                frame = cv2.resize(frame, (640, 480))  # Resize to 640x480 for faster processing
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                # Initialize feedback and score for the frame
                feedback = []
                score = 0

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    frame_feedback, frame_score, frame_timestamps = analyze_posture(landmarks, frame_time)
                    feedback_list.extend(frame_feedback)
                    total_score += frame_score
                    timestamps_list.extend(frame_timestamps)

                    # Draw landmarks on the frame
                    draw_landmarks(frame, results)

                # Display feedback on the video
                cv2.putText(frame, f"Score: {total_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                for idx, fb in enumerate(feedback):
                    cv2.putText(frame, fb, (10, 70 + (30 * idx)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Write the processed frame to output video
                out.write(frame)

            cap.release()
            out.release()

            # Display the final score and feedback
            st.write(f"Total Score: {total_score}")
            st.write("Feedback Summary:")
            for fb in set(feedback_list):
                st.write(f"- {fb}")

            # Save feedback report
            report_file = save_feedback_report(feedback_list, total_score, timestamps_list)
            st.download_button(
                label="Download Feedback Report",
                data=open(report_file).read(),
                file_name="yoga_feedback_report.txt",
                mime="text/plain",
            )

if __name__ == '__main__':
    main()
