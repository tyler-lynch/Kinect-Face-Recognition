import os
import time
import cv2
import numpy as np
import threading

# Define paths to model data
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir, 'model_data\\deploy.prototxt')
caffemodel_path = os.path.join(base_dir, 'model_data\\weights.caffemodel')

# Read the model
model1 = cv2.dnn.readNet(prototxt_path, caffemodel_path) # Model for live video
model2 = cv2.dnn.readNet(prototxt_path, caffemodel_path) # Model for reference image
model3 = cv2.dnn.readNet(prototxt_path, caffemodel_path) # Model for target image

# Capture from webcam or kinect
#cap = cv2.VideoCapture(0)   # Webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Kinect

# Image set counter for this session
session_image_sets = 0  # Number of image sets created during session
con_threshold = 0.5     # Confidence level threshold for the model

# Creates folders for reference and target images
if not os.path.exists('image_sets'):
    print("New directory created")
    os.makedirs('image_sets')


# Method for live video feed of the kinect
def video_feed():
    print('Video feed started')
    
    while True:

        # Press q to close the feed
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        # Grabs frame from the kinect feed
        ret, frame = cap.read()

        # If feed fails break
        if not ret:
            print('Failed to grab frame')
            break

        # Create blob from video frame
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Run blob into the model1 and store outputs in detections
        model1.setInput(blob)
        detections = model1.forward()

        # Itereate and draw boxes around all detected faces
        for i in range(0, detections.shape[2]):

            # Creates the box and seperates it into points
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Grab the confindence of the detection
            confidence = detections[0, 0, i, 2]

            # If confidence > con_threshold, show box around face
            if (confidence > con_threshold):
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # Displays video feed
        cv2.imshow('Kinect', frame)


# Method for grabbing the reference and target images from the video feed
def face_capture_feed():
    global session_image_sets

    print('Face gathering started')

    while True:

        # Grabs frame from the kinect feed
        ret, frame = cap.read()

        # If feed fails break
        if not ret:
            print('Failed to grab frame')
            break

        # Create blob from video frame
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Run blob into model2 and store outputs in detections
        model2.setInput(blob)
        detections = model2.forward()

        # Boolean for if a reference image was found
        ref_found = False

        # Itereate through detected faces and capture reference images
        for i in range(0, detections.shape[2]):
            
            # Creates the box and seperates it into points
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Grab the confindence of the detection
            confidence = detections[0, 0, i, 2]

            # If confidence > con_threshold, show box around face
            if (confidence > con_threshold):
                try:
                    img = frame[startY:endY, startX:endX]
                    img_name = base_dir + '\\image_sets\\{}_reference.png'.format(session_image_sets + 1)
                    cv2.imwrite(img_name, img)
                    ref_found = True
                    print('Writing reference image')
                except:
                    print('Reference failed to write')
            
        # If a reference image isn't found it will restart and find a new refernce image
        if not ref_found:
            continue
        
        # Delays for 5 seconds, then grabs a new frame from the capture and runs it on model3 to get the target image.
        time.sleep(5)

        # Grabs frame from the kinect feed
        ret2, frame2 = cap.read()

        # If feed fails break
        if not ret2:
            print('Failed to grab frame')
            break

        # Create blob from video frame
        (h2, w2) = frame2.shape[:2]
        blob2 = cv2.dnn.blobFromImage(cv2.resize(frame2, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Run blob into model3 and store outputs in detections
        model3.setInput(blob2)
        detections2 = model3.forward()
        
        # Boolean for if a target image was found
        tar_found = False

        # Itereate through each detected face and make target images
        for i in range(0, detections2.shape[2]):
            
            # Creates the box and seperates it into points
            box2 = detections2[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX2, startY2, endX2, endY2) = box2.astype('int')

            # Grab the confindence of the detection
            confidence2 = detections2[0, 0, i, 2]

            # If confidence > con_threshold, capture face as .png
            if (confidence2 > con_threshold):
                img2 = frame2[startY2:endY2, startX2:endX2]
                img_name2 = base_dir + '\\image_sets\\{}_target.png'.format(session_image_sets + 1)
                try:
                    cv2.imwrite(img_name2, img2)
                    tar_found = True
                    print('Writting target')
                except:
                    print('Target failed to write')

        # If a targt image isn't found it will restart and find a new refernce image and try to make the set again
        if not tar_found:
            continue

        # Update the image set counter
        session_image_sets += 1

# Threads
capture_thread = threading.Thread(target=face_capture_feed, daemon=True) # Face capturing thread

# Start the threads and video capture
capture_thread.start()
video_feed()

# Clean up
cap.release()
cv2.destroyAllWindows()