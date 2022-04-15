import cv2
import mediapipe as mp
import numpy as np


# Initialise holistic model For detecting face, pose and hand tracking
mediapipe_holistic = mp.solutions.holistic
# Utilities for drawing using mediapipe
mediapipe_draw = mp.solutions.drawing_utils


# Function to read image and use holistic model to detect body parts & poses
def mediapipe_detector(model, img):
    # Color Conversion
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False      # Image not writeable
    result = model.process(img)
    img.flags.writeable = True       # Image is writeable
    # Color Reconversion
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, result

# Style 2 Body Mark - Styled
def style_drawmarks(result, img): 
    # Draw pose markers
    mediapipe_draw.draw_landmarks(img, result.pose_landmarks, mediapipe_holistic.POSE_CONNECTIONS,
                             mediapipe_draw.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=4), 
                             mediapipe_draw.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2)
                             ) 
    
     # Draw face markers
    mediapipe_draw.draw_landmarks(img, result.face_landmarks, mediapipe_holistic.FACEMESH_CONTOURS, 
                             mediapipe_draw.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=2), 
                             mediapipe_draw.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=2)
                             )
    
    # Draw right hand markers  
    mediapipe_draw.draw_landmarks(img, result.right_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS, 
                             mediapipe_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mediapipe_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
    
    # Draw left hand markers
    mediapipe_draw.draw_landmarks(img, result.left_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS, 
                             mediapipe_draw.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mediapipe_draw.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )  


# Function to extract keypoints from body pose results into numpy arrays
def prep_keypoints(results):
    # Pose Features Array
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # Facial Features Array
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    # Left & Right Hand features array
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Concatenating all body pose features necessary for sign language recognition
    return np.concatenate([pose,face, left_hand, right_hand])


# Function to show prediction texts
def visualiser(res, gestures, ip_frame, colours):
    op_frame = ip_frame.copy()
    for i, j in enumerate(res):
        cv2.rectangle(op_frame, (0,60+i*40), (int(j*100), 90+i*40), colours[i], -1)
        cv2.putText(op_frame, gestures[i], (0, 85+i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return op_frame