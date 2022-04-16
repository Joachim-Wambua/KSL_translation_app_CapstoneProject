# app.py
# import the necessary packages
from multiprocessing.connection import wait
from flask import Flask, render_template, Response
from functions import *
from keras.models import load_model
import time
import mediapipe as mp
import cv2


# Importing the model
sign_lang_model = load_model('models/optimised_model.h5')

# Variables
# Logic to run Webcam for real time tests
colours = [(245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16), (16,117,245), (245,117,16)]

# Hand gestures to be detected
gestures = np.array(['apple','drink', 'hello', 'me', 'mirror', 'pipe', 'thankyou', 'time', 'woman', 'you'])


app = Flask(__name__)
@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')
def gen():
    # Logic to run Webcam for real time tests
    sentence = list()
    video_sequence = list()
    min_threshold = 0.8
    predictions = list()
    

    """Video streaming generator function."""
    vid_capture = cv2.VideoCapture(0)
    # Accessing mediapipe holistic model 
    with mediapipe_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while vid_capture.isOpened():
            # Read feed
            ret, frame = vid_capture.read()

            # Detect Realtime Objects using mediapipe
            img, result = mediapipe_detector(holistic,frame)
            # print(result)
            
            #Drawing Body Landmarks
            style_drawmarks(result, img)
            
            # Making Predictions
            # Preping datapoints from real time sequence
            data_points = prep_keypoints(result)
            # Add datapoint to vid_sequence list
            video_sequence.append(data_points)
            # Select last 30 frames from video sequence
            video_sequence = video_sequence[-30:]
            
            # From last 30 video frames...
            # if not result.left_hand_landmarks and not result.right_hand_landmarks:
            if len(video_sequence) == 30:
                # Make Predictions
                res = sign_lang_model.predict(np.expand_dims(video_sequence, axis=0))[0]
                # Print predicted result
                print(gestures[np.argmax(res)])
                predictions.append(np.argmax(res))
            
                #3. Visualise logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > min_threshold: 
                        
                        if len(sentence) > 0: 
                            if gestures[np.argmax(res)] != sentence[-1]:
                                sentence.append(gestures[np.argmax(res)])
                        else:
                            sentence.append(gestures[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                img = visualiser(res, gestures, img, colours)
                    
            # Handling Video Display
            cv2.rectangle(img, (0,0), (640, 40), (117,245,16), -1)
            cv2.putText(img, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            image = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

            # Break Gracefully
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        vid_capture.release()
        cv2.destroyAllWindows()


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)