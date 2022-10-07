import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
from imutils import paths
import argparse


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


cap = cv2.VideoCapture(0)
i=0
frame= {}

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
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
            elif -28<y<-15 and -7<x<7:
                text="You are looking 8'0 Clock, try to Face 12'0 Clock"
            elif -10<y<-0 and -14<x<-7:
                text="You are looking 7'0 Clock, try to Face 12'0 Clock"
            elif -15<x<-10:
                text="You are looking 6'0 Clock, try to Face 12'0 Clock"
            elif -10<x<-5:
                text="You are looking 5'0 Clock, try to Face 12'0 Clock"
            elif -5<x<-0.01:
                text="You are looking 4'0 Clock, try to Face 12'0 Clock"
            elif 1<y<7:
                text="You are looking 1'0 Clock, try to Face 12'0 Clock"
            elif 7<y<14:
                text="You are looking 2'0 Clock, try to Face 12'0 Clock"
            elif 14<y<21:
                text="You are looking 3'0 Clock, try to Face 12'0 Clock"
            elif -7<y<-0.01:
                text="You are looking 11'0 Clock, try to Face 12'0 Clock"
            elif -14<y<-7:
                text="You are looking 10'0 Clock, try to Face 12'0 Clock"
            elif -21<y<-14:
                text="You are looking 9'0 Clock, try to Face 12'0 Clock"
            else:
                text="Your face is not in location"
            
            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
            cv2.line(image, p1, p2, (255, 0, 0), 3)
            
            #Detecting Blur
            var=cv2.Laplacian(image, cv2.CV_64F).var()
            if var >100:
                indication="Blur"
            else:
                indication="Image is Blurred"
            
            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (550, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (550, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(image, "{}: {:.2f}".format(indication, var), (515, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
        end = time.time()
        totalTime = end - start

        #fps = 1 / totalTime
        
        #print("FPS: ", fps)
        
        #Exporting the data to a csv file
        i += 1
        frame[i]=[x, y, z, text]
        
        
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
        text = "Please Center to the Camera"
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        i += 1
        frame[i]=[x, y, z, text]
       

    cv2.imshow('Head Pose Estimation', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

dataset=pd.DataFrame.from_dict(frame, orient= 'index', columns =['X','Y','Z','FPS','Text'])
dataset.to_csv('head_pose_estimation.csv')
cap.release()