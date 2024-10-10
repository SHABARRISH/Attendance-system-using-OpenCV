import tkinter as tk
from tkinter import messagebox
import cv2
import pickle
import csv
import time
import datetime
import numpy as np
import face_recognition
import threading  # Initialize the webcam 
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load the known face encodings and their corresponding IDs
with open("EncodeFile.p", 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)

encodeListKnown, studentIds = encodeListKnownWithIds[0], encodeListKnownWithIds[1]

# Set up CSV file for logging recognized faces
output_file_path = 'C:/Users/shaba/Documents/recognized_faces.csv'  # Update this path
with open(output_file_path, mode='w', newline='') as csv_file:
    fieldnames = ['Name', 'Timestamp']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()  # Write the header to the CSV file

    # Dictionary to track the last recognized time for each person
    last_recognition_time = {name: 0 for name in studentIds}  # Initialize with 0 seconds
    recognition_cooldown = 5 * 60  # 5 minutes in seconds

    while True:
        success, img = cap.read()  # Capture a frame from the webcam
        
        if not success:
            print("Error: Could not read frame.")
            break

        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Find faces and encodings in the current frame
        faceLocations = face_recognition.face_locations(imgRgb)
        encodingsCurrentFrame = face_recognition.face_encodings(imgRgb, faceLocations)

        for encodeFace, faceLoc in zip(encodingsCurrentFrame, faceLocations):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDist)  # Find the closest match

            if matches[matchIndex]:  # If there is a match
                name = studentIds[matchIndex]  # Get the name of the matched student
                current_time = time.time()  # Get the current time in seconds

                # Draw a rectangle around the face and put the name
                y1, x2, y2, x1 = faceLoc  # Get the coordinates of the face
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle

                # Add name above the rectangle
                cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check if the person was recognized within the cooldown period
                if current_time - last_recognition_time[name] >= recognition_cooldown:
                    # Update the last recognized time
                    last_recognition_time[name] = current_time

                    # Append recognized face data to the CSV file
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Format as YYYY-MM-DD HH:MM:SS
                    writer.writerow({'Name': name, 'Timestamp': timestamp})
                    print(f"Data logged: {name} at {timestamp}")

        # Show the webcam feed without background overlay
        cv2.imshow("Face Attendance", img)  # Show the webcam image only

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows(