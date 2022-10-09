# import libraries
import tkinter as tk # build interface
import customtkinter as ck 

import pandas as pd 
import numpy as np # data transofrmation
import pickle # load up ml model

import mediapipe as mp # pose tracking
import cv2 
from PIL import Image, ImageTk 

from landmarks import landmarks # set of column names

# window for the application UI
window = tk.Tk()
window.geometry("480x700") # size of the app
window.title("GymBuddy") # name of the app
ck.set_appearance_mode("dark") # make the interface looks better

# create some labels
classLabel = ck.CTkLabel(window, height=40, width = 120, 
                                text_font=("Arial", 20), # text font and size
                                text_color="black", padx=10
                                )
classLabel.configure(text='STAGE') #what shows up in the labels
classLabel.place(x=10, y=1) # sets the position
counterLabel = ck.CTkLabel(window, height=40, width =120, 
                                text_font=("Arial", 20), # text font and size
                                text_color="black", padx=10
                                )
counterLabel.configure(text='REPS') #what shows up in the labels
counterLabel.place(x=160, y=1) # sets the position
probLabel = ck.CTkLabel(window, height=40, width = 120, 
                                text_font=("Arial", 20), # text font and size
                                text_color="black", padx=10
                                )
probLabel.configure(text='PROB') #what shows up in the labels
probLabel.place(x=300, y=1) # sets the position
#titles for our boxes
classBox = ck.CTkLabel(window, height=40, width =120, 
                                text_font=("Arial", 20), # text font and size
                                text_color="black", fg_color="blue"
                                )
classBox.configure(text='') #what shows up in the labels
classBox.place(x=10, y=41) # sets the position 
counterBox = ck.CTkLabel(window, height=40, width =120, 
                                text_font=("Arial", 20), # text font and size
                                text_color="black", fg_color="blue"
                                )
counterBox.configure(text='') #what shows up in the labels
counterBox.place(x=160, y=41) # sets the position 
probBox = ck.CTkLabel(window, height=40, width = 120, 
                                text_font=("Arial", 20), # text font and size
                                text_color="black", fg_color="blue"
                                )
probBox.configure(text='') #what shows up in the labels
probBox.place(x=300, y=41) # sets the position

# the above files will be updated dynamically once the ml model is executed.

# button to reset the counter
def reset_counter():
    global counter
    counter=0

button = ck.CTkButton(window, text='Reset', # name of the button
                                command=reset_counter, # function it needs to trigger
                                height=40, width = 120, 
                                text_font=("Arial", 20), # text font and size
                                text_color="black", fg_color="red"
                                )
button.place(x=10, y=600)


# frame window, this is where the captured image will be shown from webcam
frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90)
lmain = tk.Label(frame) # updating label
lmain.place(x=0, y=0)

# import mediapipe
mp_drawing = mp.solutions.drawing_utils # drawing utilities
mp_pose = mp.solutions.pose # pose estimation model
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# import deadlift model
with open('deadlift.pkl', 'rb') as f: # 'rb' -> read binary files
    model = pickle.load(f)

# get video capture
cap = cv2.VideoCapture(0)

# variables
current_stage = '' # it tells whether we are up or down in deadlift
counter = 0 # how many deadlifts you have done
bodylang_prob = np.array([0,0]) # probability of each of the poses
bodylang_class = '' 

# create a detect function (running in loop)
def detect():
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob

    # capture the frame
    ret, frame = cap.read()
    # convert from bgr to rgb
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # process the image by passing to pose estimation model
    results = pose.process(image)
    #draw the poses
    mp_drawing.draw_landmarks(  
                                image, # pass the image
                                results.pose_landmarks,  
                                mp_pose.POSE_CONNECTIONS, 
                                mp_drawing.DrawingSpec(
                                                        color=(0,0,255), 
                                                        thickness=2, 
                                                        circle_radius=2),
                                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
    # this is where we are going to do the detection
    try:
        row= np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        X = pd.DataFrame([row], columns= landmarks)
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0]

        if bodylang_class =="down" and bodylang_prob[bodylang_prob.argmax()] > 0.7 :
            current_stage = "down"
        elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7 :
            current_stage = "up"
            counter += 1
    except Exception as e:
        print(e)
    #update our image
    img = image[:, :460, :] # slice the image and grab 460 pixels
    imgarr = Image.fromarray(img) # convert the img to array (using pillow)
    imgtk = ImageTk.PhotoImage(imgarr) # convert the array to usable img
    # pass through lmain
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)

    counterBox.configure(text=counter)
    probBox.configure(text=bodylang_prob[bodylang_prob.argmax()])
    classBox.configure(text=current_stage)
detect()
window.mainloop() # start the app
