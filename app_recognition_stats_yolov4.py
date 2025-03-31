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

results_filename = "results_yolo.txt"

database_path = "/home/metrine/Desktop/myfiles/APP_NEW/database"
images_path = "/home/metrine/Desktop/myfiles/APP_NEW/images"
results_folder_path = "/home/metrine/Desktop/myfiles/APP_NEW/results"

# Define the full file path
results_file_path = os.path.join(results_folder_path, results_filename)

cam_select = "norm" #"reco"

owner_password = "pass"

train_model = False
train_name = ""
train_count = 0

frame_count = 0
count_match = 0
count_mismatch = 0
count_unknown = 0

# Application ********************************** #

class WebcamApp:

    def __init__(self, window):
        
        global resH, resV, database_path, cam_select, results_file_path

        # Initialize the main Tkinter window
        self.window = window
        self.window.title("APP FACE RECOGNITION")
        
        # Create Frame 1 for the webcam feed (Video)
        self.winFrame1 = tk.Frame(window)
        self.winFrame1.pack(side="top")
        
        # Create Frame 2 for the buttons
        self.winFrame2 = tk.Frame(window)
        self.winFrame2.pack(side="bottom")
        
        # Create results file
        self.create_result_file(results_file_path)
        
        # Load Face Encodings from Database
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces(database_path)
        
        # Load Yolo
        self.load_yolo_tiny()

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
        
        # Initialize countdown variable
        self.countdown_value = 5 #Five Seconds
        
        # Label to display countdown in Frame1
        self.countdown_label = Label(self.winFrame1, text=str(self.countdown_value), font=("Arial", 30, "bold"), fg="red", bg="black")
        self.countdown_label.place(relx=0.5, rely=0.1, anchor="center")  # Center top of webcam frame

        # Start countdown overlay
        self.start_countdown()
        
        # Update variable to run in "reco" mode
        self.window.after(5000, self.update_selection)
        
        # Process data after 15 seconds
        self.window.after(15000, self.calculate_values)
        #self.window.after(20000, self.calculate_recall)
        #self.window.after(15000, self.calculate_precision)
        
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
        self.button_runtest = tk.Button(self.winFrame2, text="RUN TEST", command=self.rerun_test)
        
        self.button_close = tk.Button(self.winFrame2, text="CLOSE APP", command=self.close_application)
        self.button_submit_close = tk.Button(self.winFrame2, text="SUBMIT", command=self.check_close_password)
        self.button_exit_close = tk.Button(self.winFrame2, text="EXIT CLOSE", command=self.exit_close_application)
        
        # Add Start-up Buttons in a Grid Layout within Frame2
        
        self.button_train.grid(row=0, column=0, padx=5, pady=5)
        self.button_run.grid(row=0, column=1, padx=5, pady=5)   
        self.button_capture.grid(row=0, column=2, padx=5, pady=5)
        self.button_close.grid(row=0, column=3, padx=5, pady=5)
        self.button_runtest.grid(row=0, column=4, padx=5, pady=5)
        
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
    
    def load_yolo_tiny(self):
        
        # Load YOLOv4-Tiny Model
        self.yolo_net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
        self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        #self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get YOLO output layers
        self.layer_names = self.yolo_net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
        
        return self.yolo_net, self.layer_names, self.output_layers
    
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
        
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
        self.yolo_net.setInput(blob)
        detections = self.yolo_net.forward(self.output_layers)

        face_locations = []
        confidences = []
        boxes = []
        
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                confidence = max(scores)
                if confidence > 0.25:
                    center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype("int")
                    
                    # Adjust box to make it tighter
                    reduction_factor = 0.25  # Adjust this value to control box size
                    w = int(w * (1 - reduction_factor))  
                    h = int(h * (1 - reduction_factor + 0.1))
                    
                    x = max(int(center_x - w / 2), 0)
                    y = max(int(center_y - h / 2), 0)
                    right = min(x + w, width)
                    bottom = min(y + h, height)
                    
                    # Store box and confidence score
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.25, nms_threshold=0.25)

        # Only keep the best detection
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                right = x + w
                bottom = y + h
                face_locations.append((y, right, bottom, x))  # Format for face_recognition
        
        face_encodings = [self.extract_face_encoding(frame, loc) for loc in face_locations]

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            #name = "Unknown"

            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
                count_match += 1
            else:
                name = "Unknown"
                count_unknown += 1

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert Frame to RGB and Display in App
        self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image=self.current_image)
        self.webcam_label.imgtk = self.photo  # Keep a reference to prevent garbage collection
        self.webcam_label.configure(image=self.photo)
        self.window.after(5, self.update_webcam_reco) # Update every 5ms
        
    # Run Camera WITH or WITHOUT Face Detection, depending on selection
    def update_webcam(self):
        
        global tolerance, train_model, train_count, frame_count, count_match, count_mismatch, count_unknown

        frame = self.picam.capture_array()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        self.current_frame = frame
        #frame_count = frame_count + 1
        
        if cam_select == "reco":
            
            frame_count = frame_count + 1
            
            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
            self.yolo_net.setInput(blob)
            detections = self.yolo_net.forward(self.output_layers)

            face_locations = []
            confidences = []
            boxes = []
            
            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    confidence = max(scores)
                    if confidence > 0.25:
                        center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype("int")
                        
                        # Adjust box to make it tighter
                        reduction_factor = 0.25  # Adjust this value to control box size
                        w = int(w * (1 - reduction_factor))  
                        h = int(h * (1 - reduction_factor + 0.1))
                        
                        x = max(int(center_x - w / 2), 0)
                        y = max(int(center_y - h / 2), 0)
                        right = min(x + w, width)
                        bottom = min(y + h, height)
                        
                        # Store box and confidence score
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))

            # Apply Non-Maximum Suppression (NMS)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.25, nms_threshold=0.25)

            # Only keep the best detection
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    right = x + w
                    bottom = y + h
                    face_locations.append((y, right, bottom, x))  # Format for face_recognition
            
            face_encodings = [self.extract_face_encoding(frame, loc) for loc in face_locations]

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                #name = "Unknown"

                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    count_match += 1
                else:
                    name = "Unknown"
                    count_unknown += 1

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
            
        if frame_count > 10000:
            frame_count = 0

    def extract_face_encoding(self, frame, face_location):
        """Extracts face encoding from detected faces."""
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(face_image)

        return encodings[0] if encodings else np.zeros((128,))
    
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
    
    def create_result_file(self, resultspath):
        
        # Check if file exists before creating it
        if not os.path.exists(resultspath):
            with open(resultspath, "w") as file:
                file.write("Precision\tRecall\n")  # Initial header
                file.write("=" * 30 + "\n")  # Separator
            print(f"File created: {resultspath}")
        else:
            print(f"File already exists: {resultspath}")

        
    
    def start_countdown(self):
        """Starts a 5-second countdown overlay on the webcam feed."""
        if self.countdown_value > 0:
            self.countdown_label.config(text=str(self.countdown_value))
            self.countdown_value -= 1
            self.window.after(1000, self.start_countdown)  # Run every second
        else:
            self.countdown_label.place_forget()  # Remove the label from UI
            #self.countdown_label.config(text="")  # Hide countdown after 5 seconds
    
    def update_selection(self):
        
        global cam_select, frame_count, count_match, count_mismatch, count_unknown
        
        frame_count = 0
        count_match = 0
        count_mismatch = 0
        count_unknown = 0
        
        cam_select = "reco"
        
    def rerun_test(self):
        
        self.countdown_value = 5 #Five Seconds
        self.start_countdown()
        self.window.after(5000, self.update_selection)
        self.window.after(15000, self.calculate_values)  # Process data after 15 seconds
        
        
    def calculate_precision(self):
        #TP/(TP + FP) : M/(M + U)
        
        global cam_select, frame_count, count_match, count_mismatch, count_unknown
        
        cam_select = "norm"
        
        self.blank = frame_count - count_match - count_unknown
        self.precision = count_match/(count_match + (count_unknown + self.blank))
        self.precision = round(self.precision, 6)
        
        self.countdown_label.place(relx=0.5, rely=0.1, anchor="center")  # Center top of webcam frame
        self.countdown_label.config(text=f"Precision Value: {self.precision}")
        print("\nPrecision Result: ")
        print(self.precision)
    
    def calculate_recall(self):
        #TP/(TP + FN) : M/(M + U)
        
        global cam_select, frame_count, count_match, count_mismatch, count_unknown
        
        cam_select = "norm"
        
        self.recall = count_match/(count_match + count_unknown)
        self.recall = round(self.recall, 6)
        
        self.countdown_label.place(relx=0.5, rely=0.1, anchor="center")  # Center top of webcam frame
        self.countdown_label.config(text=f"Recall Value: {self.recall}")
        #print("\nRecall Result: ")
        #print(self.recall)
        
    def calculate_values(self):
        #TP/(TP + FP) : M/(M + U)
        
        global cam_select, frame_count, count_match, count_mismatch, count_unknown, results_file_path
        
        cam_select = "norm"
        
        self.blank = frame_count - count_match - count_unknown
        self.precision = count_match/(count_match + (count_unknown + self.blank))
        self.precision = round(self.precision, 6)
        
        self.recall = count_match/(count_match + count_unknown)
        self.recall = round(self.recall, 6)
        
        self.countdown_label.place(relx=0.5, rely=0.1, anchor="center")  # Center top of webcam frame
        self.countdown_label.config(text=f"Precision Value: {self.precision} \nRecall Value: {self.recall}")
        print(f"Precision Result: {self.precision}  Recall Result: {self.recall}")
        
        with open(results_file_path, "a") as file:
            file.write(f"{self.precision:.6f}\t{self.recall:.6f}\n")
            #file.write(f"Precision Result: {self.precision}  Recall Result: {self.recall}\n")
            #file.write("-" * 30 + "\n")  # Separator for readability



# Tkinter window
mygui = tk.Tk()

# Create webcam App
app = WebcamApp(mygui)

# Run Loop
mygui.mainloop()