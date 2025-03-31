import os
import tkinter as tk
from tkinter import Label
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from picamera2 import Picamera2
import face_recognition

# Global Variables ***************************** #
num = 0
tolerance = 0.5     # Adjust the tolerance value as needed
# Notes:
        # HD is 720p = 1280 by 720 (H x V) 16:9
        # SD is 480p = 854 by 480 (H x V) 16:9
        # SD is 480p = 640 by 480 (H x V) 4:3
resH = 800
resV = 480

database_path = "/home/metrine/Desktop/myfiles/APP_NEW/database"
images_path = "/home/metrine/Desktop/myfiles/APP_NEW/images"

cam_select = "reco"

owner_password = "pass"

train_model = False
train_name = ""
train_count = 0

# Application ********************************** #

class WebcamApp:

    def __init__(self, window):
        
        global resH, resV, database_path, cam_select

        # Initialize the main Tkinter window
        self.window = window
        self.window.title("APP FACE RECOGNITION")
        
        # Create Frame 1 for the webcam feed (Video)
        self.winFrame1 = tk.Frame(window)
        self.winFrame1.pack(side="top")
        
        # Create Frame 2 for the buttons
        self.winFrame2 = tk.Frame(window)
        self.winFrame2.pack(side="bottom")
        
        # Load Face Encodings from Database
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces(database_path)

        # Create Camera Object, configure and start it
        self.picam = Picamera2(0)        
        self.picam.configure(self.picam.create_preview_configuration(main={"format": 'RGB888', "size": (resH, resV)}))
        self.picam.start()

        # Create and Initialize image and frame variables
        self.current_image = None
        self.current_frame = None
        
        # Create and Add Label to display image frames
        self.webcam_label = Label(self.winFrame1)
        self.webcam_label.pack()
        
        # Capture frames and display to Frame1
        self.update_webcam()
        # self.update_webcam_norm()
        # self.update_webcam_reco()
        
        # Create Buttons, Text and Labels for Frame2
        
        self.button_train = tk.Button(self.winFrame2, text="TRAIN", command=self.start_train_model)
        self.button_submit_pass = tk.Button(self.winFrame2, text="SUBMIT", command=self.check_train_password)
        self.button_submit_name = tk.Button(self.winFrame2, text="USE NAME", command=self.get_train_name)
        self.button_exit_train = tk.Button(self.winFrame2, text="EXIT TRAIN", command=self.exit_train_model)
        
        self.entry_password = tk.Entry(self.winFrame2, show="*", width=15)
        self.entry_name = tk.Entry(self.winFrame2, width=15)
        
        self.label_password = tk.Label(self.winFrame2, text="Enter Password: ", font=("Arial", 12))
        self.label_name = tk.Label(self.winFrame2, text="Enter Your Name: ", font=("Arial", 12))
        self.label_training = tk.Label(self.winFrame2, text="Now Training...", font=("Arial", 12))   
        
        self.button_run = tk.Button(self.winFrame2, text="RUN", command=self.run_reco)   
        self.button_capture = tk.Button(self.winFrame2, text="CAPTURE", command=self.capture_image)
        self.button_delete = tk.Button(self.winFrame2, text="DELETE", command=self.delete_image)
        
        self.button_close = tk.Button(self.winFrame2, text="CLOSE APP", command=self.close_application)
        self.button_submit_close = tk.Button(self.winFrame2, text="SUBMIT", command=self.check_close_password)
        self.button_exit_close = tk.Button(self.winFrame2, text="EXIT CLOSE", command=self.exit_close_application)
        
        # Add Start-up Buttons in a Grid Layout within Frame2
        
        self.button_train.grid(row=0, column=0, padx=5, pady=5)
        self.button_run.grid(row=0, column=1, padx=5, pady=5)   
        self.button_capture.grid(row=0, column=2, padx=5, pady=5)
        self.button_close.grid(row=0, column=3, padx=5, pady=5)
        
        # self.entry_password.grid(row=0, column=1)
        # self.entry_password.grid_forget()   # Do not dispay
        # self.entry_name.grid(row=0, column=2)
        # self.entry_name.grid_forget()   # Do not dispay
        # self.label_training.grid(row=0, column=3, padx=5, pady=5)
        # self.label_training.grid_forget()   # Do not dispay
        # self.button_submit_pass.grid(row=0, column=4, padx=5, pady=5)
        # self.button_submit_pass.grid_forget()    # Do not dispay
        # self.button_submit_name.grid(row=0, column=5, padx=5, pady=5)
        # self.button_submit_name.grid_forget()    # Do not dispay
        
    
    # Load Images from Folder and Create Face Encodings
    def load_known_faces(self, folder_path):
    
        # known_face_encodings = []   # Moved to Init
        # known_face_names = []       # Moved to Init
    
        for filename in os.listdir(folder_path):

            if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                image_path = os.path.join(folder_path, filename)
                self.known_face_names.append(os.path.splitext(filename)[0])

                # Image in a format that face_recognition can detect faces
                image = face_recognition.load_image_file(image_path)

                face_encodings = face_recognition.face_encodings(image)
    
                if face_encodings:
                    face_encoding = face_encodings[0]
                    self.known_face_encodings.append(face_encoding)
    
        return self.known_face_encodings, self.known_face_names
        
    # Run Camera WITHOUT Face Detection
    def update_webcam_norm(self):
        
        global tolerance

        frame = self.picam.capture_array()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        self.current_frame = frame

        # Convert Frame to RGB and Display in App
        self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image=self.current_image)
        self.webcam_label.imgtk = self.photo  # Keep a reference to prevent garbage collection
        self.webcam_label.configure(image=self.photo)
        self.window.after(5, self.update_webcam_norm) # Update every 5ms
    
    # Run Camera WITH Face Detection
    def update_webcam_reco(self):
        
        global tolerance

        frame = self.picam.capture_array()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        self.current_frame = frame
        
        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(frame)     # List of face locations
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            # Check if the face matches any known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)   
            name = "Unknown"

            # If a match was found, use the name of the known face
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
                namearray = name.split()
                name = namearray[0]         # Extract only the name
            
            #if name == "Unknown":
                #send_email(frame, server)

            # Draw a rectangle and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.9, (255, 255, 0), 1)

        # Convert Frame to RGB and Display in App
        self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image=self.current_image)
        self.webcam_label.imgtk = self.photo  # Keep a reference to prevent garbage collection
        self.webcam_label.configure(image=self.photo)
        self.window.after(5, self.update_webcam_reco) # Update every 5ms
        
    # Run Camera WITH or WITHOUT Face Detection, depending on selection
    def update_webcam(self):
        
        global tolerance, train_model, train_count

        frame = self.picam.capture_array()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        self.current_frame = frame
        
        if cam_select == "reco":
            
            # Find all face locations and face encodings in the current frame
            face_locations = face_recognition.face_locations(frame)     # List of face locations
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

                # Check if the face matches any known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)   
                name = "Unknown"

                # If a match was found, use the name of the known face
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    namearray = name.split()
                    name = namearray[0]         # Extract only the name
                
                #if name == "Unknown":
                    #send_email(frame, server)

                # Draw a rectangle and display the name
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 1)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.9, (255, 255, 0), 1)

            # Convert Frame to RGB and Display in App
            self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=self.current_image)
            self.webcam_label.imgtk = self.photo  # Keep a reference to prevent garbage collection
            self.webcam_label.configure(image=self.photo)
            self.window.after(5, self.update_webcam) # Update every 5ms
            
        if cam_select == "norm":
            
            # Convert Frame to RGB and Display in App
            self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=self.current_image)
            self.webcam_label.imgtk = self.photo  # Keep a reference to prevent garbage collection
            self.webcam_label.configure(image=self.photo)
            #self.window.after(5, self.update_webcam) # Update every 5ms
            
            if train_model == True:
                
                train_count = train_count + 1
                
                if train_count >= 20:
                    train_count = 0
                    self.capture_train_image()
            
            self.window.after(5, self.update_webcam) # Update every 5ms

    
    def run_norm(self):
        
        global cam_select
        cam_select = "norm"
    
    
    def run_reco(self):
        
        global cam_select
        cam_select = "reco"
        
    
    def start_train_model(self):
    
        global cam_select
        
        self.button_train.grid_forget()
        self.button_run.grid_forget()
        self.button_capture.grid_forget()
        self.button_close.grid_forget()
        
        self.label_password.grid(row=0, column=1, padx=5, pady=5)
        self.entry_password.grid(row=0, column=2)
        self.button_submit_pass.grid(row=0, column=3, padx=5, pady=5)
        self.button_exit_train.grid(row=0, column=4, padx=5, pady=5)
        
        messagebox.showinfo("Access Required", "Enter Password to Train Model")
    
    
    def check_train_password(self):
        
        global cam_select
        
        entered_password = self.entry_password.get()
        if entered_password == owner_password:
            
            cam_select = "norm"
            
            self.label_password.grid_forget()
            self.entry_password.grid_forget()
            self.button_submit_pass.grid_forget()
            self.button_exit_train.grid_forget()
            
            self.label_name.grid(row=0, column=1, padx=5, pady=5)
            self.entry_name.grid(row=0, column=2)
            self.button_submit_name.grid(row=0, column=3, padx=5, pady=5)
            self.button_exit_train.grid(row=0, column=4, padx=5, pady=5)
            
            messagebox.showinfo("Verified !", "Enter Your First Name")
            
        else:
            
            cam_select = "reco"
            
            self.label_password.grid_forget()
            self.entry_password.grid_forget()
            self.button_submit_pass.grid_forget()
            self.button_exit_train.grid_forget()
            
            self.button_train.grid(row=0, column=0, padx=5, pady=5)
            self.button_run.grid(row=0, column=1, padx=5, pady=5)
            self.button_capture.grid(row=0, column=2, padx=5, pady=5)
            self.button_close.grid(row=0, column=3, padx=5, pady=5)
            
            messagebox.showerror("Access Denied", "Incorrect Password Entered!")
            
    
    def get_train_name(self):
        
        global train_model, train_name, num
        
        entered_name = self.entry_name.get()
        if entered_name:  # Check that entry is not empty
            
            name_exists = self.check_database_names(entered_name)
            
            if name_exists == False:
                result = messagebox.showinfo("Name Check", "Welcome " + entered_name + ". Press OK to start Training your Model!")
                train_name = entered_name
                # Set train_model to true after waiting for OK pressed
                while True:
                    if result:
                        train_model = True  # Allow training to start
                        break
            else:
                 messagebox.showwarning("Name Check", "Entered name already exists! Please enter a unique name.")
                 
        else:
            messagebox.showwarning("Name Check", "No name entered! Please enter a your name.")
    
    
    def exit_train_model(self):
        
        global cam_select
        
        cam_select = "reco"
        
        # self.label_password.grid_forget()
        # self.entry_password.grid_forget()
        # self.button_submit_pass.grid_forget()
        
        self.label_name.grid_forget()
        self.entry_name.grid_forget()
        self.button_submit_name.grid_forget()
        
        self.button_exit_train.grid_forget()
        
        self.button_train.grid(row=0, column=0, padx=5, pady=5)
        self.button_run.grid(row=0, column=1, padx=5, pady=5)
        self.button_capture.grid(row=0, column=2, padx=5, pady=5)
        self.button_close.grid(row=0, column=3, padx=5, pady=5)
    
    
    def close_application(self):
        
        self.button_train.grid_forget()
        self.button_run.grid_forget()
        self.button_capture.grid_forget()
        self.button_close.grid_forget()
        
        self.label_password.grid(row=0, column=1, padx=5, pady=5)
        self.entry_password.grid(row=0, column=2)
        self.button_submit_close.grid(row=0, column=3, padx=5, pady=5)
        self.button_exit_close.grid(row=0, column=4, padx=5, pady=5)
        
        messagebox.showinfo("Access Required", "Authorization is required to close this window.")
    
    def check_close_password(self):
        
        global cam_select
        
        entered_password = self.entry_password.get()
        if entered_password == owner_password:
            
            # Close App
            self.window.destroy()
            
        else:
            
            cam_select = "reco"
            
            self.label_password.grid_forget()
            self.entry_password.grid_forget()
            self.button_submit_close.grid_forget()
            self.button_exit_close.grid_forget()
            
            self.button_train.grid(row=0, column=0, padx=5, pady=5)
            self.button_run.grid(row=0, column=1, padx=5, pady=5)
            self.button_capture.grid(row=0, column=2, padx=5, pady=5)
            self.button_close.grid(row=0, column=3, padx=5, pady=5)
            
            messagebox.showerror("Unauthorized", "You're NOT Authorized to Close this Window!")
    
    
    def exit_close_application(self):
        
        global cam_select
        
        cam_select = "reco"
            
        self.label_password.grid_forget()
        self.entry_password.grid_forget()
        self.button_submit_close.grid_forget()
        self.button_exit_close.grid_forget()
        
        self.button_train.grid(row=0, column=0, padx=5, pady=5)
        self.button_run.grid(row=0, column=1, padx=5, pady=5)
        self.button_capture.grid(row=0, column=2, padx=5, pady=5)
        self.button_close.grid(row=0, column=3, padx=5, pady=5)
    
    # Capture an image and save to database
    def capture_train_image(self):

        global num, database_path, images_path, train_model, train_name
        
        num = num + 1

        if self.current_image is not None:
            cv2.imwrite('{}/{} {}.{}'.format(images_path, train_name, num, 'jpg'), self.current_frame)
                
        if num >= 6:
            
            train_model = False
            num = 0
            train_name = ""
            result = messagebox.showinfo("Training is Done!", "You have Successfully completed the Training! Press OK to Exit.")
            while True:
                if result:
                    self.known_face_encodings = []  # Clear first then 
                    self.known_face_names = []
                    self.load_known_faces(images_path) # Reload updated Database
                    self.exit_train_model()
                    break
            
    # Capture an image and save to file
    def capture_image(self):

        global num, database_path, images_path
        
        num = num + 1

        if self.current_image is not None:
            cv2.imwrite('{}/{} {}.{}'.format(images_path, 'capture', num, 'jpg'), self.current_frame)
            
    # Delete saved image from file
    def delete_image(self):

        global num, database_path, images_path

        if num > 0:
            os.remove('{}/{} {}.{}'.format(images_path, 'capture', num, 'jpg'))
            num = num - 1
    
    # Check if train name already exists in the database
    def check_database_names(self, filename):
        
        global images_path
        
        for file in os.listdir(images_path):
            trained_names_array = file.split()
            trained_name = trained_names_array[0]         # Extract only the name
            if trained_name == filename:
                return True
        return False

# Tkinter window
mygui = tk.Tk()

# Create webcam App
app = WebcamApp(mygui)

# Run Loop
mygui.mainloop()