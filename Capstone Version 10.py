import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import imutils
from imutils import paths
from imutils.video import VideoStream
from imutils.video import FPS
import os
import logging
import pyttsx3
import turtle


screen = turtle.Screen()    # Loading Canvas
screen.bgcolor("black")     # Bg
screen.setup(width=600, height=600)
screen.title("Analog Clock")
screen.tracer(0)

t = turtle.Turtle()
#t.hideturtle() # Make the turtle invisible
t.speed(0)  # Setting the speed to 0
t.pensize(20)    # Setting the pensize to 3


ASSETS_PATH = 'assets/'
MODEL_PATH = os.path.join(ASSETS_PATH, 'frozen_inference_graph.pb')
CONFIG_PATH = os.path.join(ASSETS_PATH, 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
LABELS_PATH = os.path.join(ASSETS_PATH, 'labels.txt')
SCORE_THRESHOLD = 0.4
NETWORK_INPUT_SIZE = (300, 300)
NETWORK_SCALE_FACTOR = 1

engine= pyttsx3.init()
rate=engine.getProperty("rate")
voices=engine.getProperty("voices")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

logger = logging.getLogger('detector')
logging.basicConfig(level=logging.INFO)

# Reading coco labels
with open(LABELS_PATH, 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')
logger.info(f'Available labels: \n{labels}\n')
COLORS = np.random.uniform(0, 255, size=(len(labels), 3))

# Loading model from file
logger.info('Loading model from tensorflow...')
ssd_net = cv2.dnn.readNetFromTensorflow(model=MODEL_PATH, config=CONFIG_PATH)

def Draw_Clock(hourr, minutee, secondd, t):

    t.up()  # not ready to draw
    t.goto(0, 210)  # positioning the turtle
    t.setheading(180)   # setting the heading to 180
    t.color("red")  # setting the color of the pen to red
    t.pendown()     # starting to draw
    t.circle(210)    # a circle with the radius 210

    t.up()  # not ready to draw
    t.goto(0, 0)    # positioning the turtle
    t.setheading(90)    # same as seth(90) in newer version

    for z in range(12):     # loop
        t.fd(190)   # moving forward at 190 units
        t.pendown()     # starting to draw
        t.fd(20)    # forward at 20
        t.penup()   # not ready to draw
        t.goto(0, 0)    # positioning the turtle
        t.rt(30)    # right at an angle of 30 degrees

    hands = [("white", 200, 20), ("white", 1, 1), ("white", 1, 1)]     # the color and the hands set
    time_set = (hourr, minutee, secondd)  # setting the time

    for hand in hands:
        time_part = time_set[hands.index(hand)]
        angle = (time_part/hand[2])*360     # setting the angle for the clock
        t.penup()   # not ready to draw
        t.goto(0, 0)    # positioning the turtle
        t.color(hand[0])    # setting the color of the hand
        t.setheading(90)    # same as seth(90)
        t.rt(angle)     # right at an angle of "right"
        t.pendown()     # ready to draw
        t.fd(hand[1])   # forward at a unit of 1st index of the hand var

cap = VideoStream(src=0).start()
i=0
frame= {}
fps = FPS().start()


while True:
    success = cap.read()
    detect= success
    image= success
    

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    detect = imutils.resize(detect, width=650)
    height, width, channels = detect.shape

    # Converting frames to blobs using mean standardization
    blob = cv2.dnn.blobFromImage(image=detect,
                                 scalefactor=NETWORK_SCALE_FACTOR,
                                 size=NETWORK_INPUT_SIZE,
                                 mean=(127.5, 127.5, 127.5),
                                 crop=False)

    # Passing blob through neural network
    ssd_net.setInput(blob)
    network_output = ssd_net.forward()

    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    engine.setProperty("rate",100)
    engine.setProperty("voice",voices[1].id)



    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)  
                
            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
          
            
            # See where the user's head tilting
            if 0<y<1:
                text="You are looking 12'0 Clock, Perfect"
                dial=0
            elif -28<y<-15 and -7<x<7:
                text="You are looking 8'0 Clock, try to Face 12'0 Clock"
                dial=13
            elif -10<y<-0 and -14<x<-7:
                text="You are looking 7'0 Clock, try to Face 12'0 Clock"
                dial=12
            elif -15<x<-10:
                text="You are looking 6'0 Clock, try to Face 12'0 Clock"
                dial=10
            elif -10<x<-5:
                text="You are looking 5'0 Clock, try to Face 12'0 Clock"
                dial=8
            elif -5<x<-0.01:
                text="You are looking 4'0 Clock, try to Face 12'0 Clock"
                dial=7
            elif 1<y<7:
                text="You are looking 1'0 Clock, Clear and Accurate"
                dial=2
            elif 7<y<14:
                text="You are looking 2'0 Clock, try to Face 12'0 Clock"
                dial=3
            elif 14<y<21:
                text="You are looking 3'0 Clock, try to Face 12'0 Clock"
                dial=5
            elif -7<y<-0.01:
                text="You are looking 11'0 Clock, Clear and Accurate"
                dial=18
            elif -14<y<-7:
                text="You are looking 10'0 Clock, try to Face 12'0 Clock"
                dial=17
            elif -21<y<-14:
                text="You are looking 9'0 Clock, try to Face 12'0 Clock"
                dial=15
            else:
                text="Your face is not in location"
                dial=0
            
            hourr = int(dial)
            minutee = int(dial)
            secondd = int(dial)

            Draw_Clock(hourr, minutee, secondd, t)
            screen.update()
            #time.sleep(1)
            t.clear()
            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
            cv2.line(image, p1, p2, (255, 0, 0), 3)
            
            #Detecting Blur
            var=cv2.Laplacian(image, cv2.CV_64F).var()
            if var >100:
                indication="You are clearly visible"
                engine.say(indication)
            else:
                indication="Image is Blurred"
                engine.say(indication)
            
            # Looping over detections
            for detection in network_output[0, 0]:
                score = float(detection[2])
                class_index = np.int(detection[1])
                label = f'{labels[class_index]}: {score:.2%}'
                
                # Drawing likely detections
                if score > SCORE_THRESHOLD:
                    left = np.int(detection[3] * width)
                    top = np.int(detection[4] * height)
                    right = np.int(detection[5] * width)
                    bottom = np.int(detection[6] * height)

                    cv2.rectangle(img=detect, rec=(left, top, right, bottom), color=COLORS[class_index], thickness=4, lineType=cv2.LINE_AA)
                    cv2.putText(img=detect, text=label, org=(left, np.int(top*0.9)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=COLORS[class_index], thickness=2, lineType=cv2.LINE_AA)
                    

            cv2.imshow("Detector", detect)
    


            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (550, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (550, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(image, "{}: {:.2f}".format(indication, var), (515, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            engine.say(text)    
            engine.runAndWait()
            
            
        end = time.time()
        totalTime = end - start

        #fps = 1 / totalTime
        
        #print("FPS: ", fps)
        
        #Exporting the data to a csv file
        i += 1
        #frame[i]=[x, y, z, text]
        
        
        #cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        # Save Frame by Frame into disk using imwrite method
        #cv2.imwrite("Frame" + str(i) + '.jpg', image)
        #i += 1
        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
    else:
        text = "Please Face towards the Camera"
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        i += 1
        #frame[i]=[x, y, z, text]
       

    cv2.imshow('Face Clock', image)
    #engine.say(text)

    
    #engine.say(text)
       
    #engine.say(indication)
    
    if  cv2.waitKey(5000):
        break
    fps.update()

    

dataset=pd.DataFrame.from_dict(frame, orient= 'index', columns =['X','Y','Z','Text'])
dataset.to_csv('head_pose_estimation.csv')
fps.stop()
logger.info(f'\nElapsed time: {fps.elapsed() :.2f}')
logger.info(f' Approx. FPS: {fps.fps():.2f}')
cv2.destroyAllWindows()
cap.stop()

