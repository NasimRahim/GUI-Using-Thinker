from tkinter import *
import cv2
import csv
import subprocess
import os
import pickle   
from tkinter import messagebox
from datetime import datetime
import cv2
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score



#_________________________________________________________________________________________________________________________________________________________
#recognition 
# Load the haarcascade classifier for face detection
cascade_path = "fyp\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load the pre-trained LBPH model
lbph_model = cv2.face.LBPHFaceRecognizer_create()
lbph_model.read("lbph_modelFYP1.yml")

# Load the trained SVM classifier (recognizer)
with open('modelALL\svm_modelFYPHaar.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Create a dictionary to map label numbers to names
label_names = {
    1: "nasim",
    2: "khai",
    3: "hariz",
    16: "mirun",
    20: "tuan arif"
}

# Create a function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces

# Create a function to recognize faces in an image
# Create a function to recognize faces in an image
def recognize_faces(image, gray):
    faces = detect_faces(image)
    predictions = []
    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        face_region = cv2.resize(face_region, (100, 100))
        label, _ = lbph_model.predict(face_region)
        predictions.append(label)
    return predictions


# Function to open a video file using file dialog
def open_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
    if file_path:
        process_video(file_path)
        messagebox.showinfo("Take Attendance","Attendance Has Been Taken")

recorded_names = set()
# Function to process the video file
def process_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    with open('attendance.csv', mode='a', newline='') as file:
        csv_writer = csv.writer(file)   

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            predictions = recognize_faces(frame, gray)  # Pass the gray image here
            faces = detect_faces(frame)
            for (x, y, w, h), label in zip(faces, predictions):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label_name = label_names.get(label, "Unknown")
                cv2.putText(frame, label_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                #enter time stamp
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                #if unknown xyah letak
                if label_name != "Unknown":
                #check if name dah ada belum
                    if label_name not in recorded_names:
                        # Write the label and timestamp to the CSV file
                        csv_writer.writerow([label_name, timestamp])
                        recorded_names.add(label_name)
                


            # Show the frame with face recognition results
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

#_________________________________________________________________________________________________________________________________________________________


#_________________________________________________________________________________________________________________________________________________________
#update model
def train():
    # Set variable to call the path from the folder untuk dataset
    dataset_path = "C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\datasetBase"

    # Load the haarcascade classifier for face detection
    cascade_path = "fyp\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Define LBPH parameters
    radius = 1
    neighbors = 8
    grid_x = 8
    grid_y = 8
    threshold = 100

    # Create an LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=radius, neighbors=neighbors, grid_x=grid_x, grid_y=grid_y, threshold=threshold)

    # Define a function to read the images from the dataset folder
    def get_images_and_labels(dataset_path):
        image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
        images = []
        labels = []
        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                if len(faces) == 0:
                    os.remove(image_path)
                for (x, y, w, h) in faces:
                    images.append(cv2.resize(gray[y:y+h, x:x+w], (100, 100)))
                    labels.append(int(os.path.split(image_path)[-1].split("_")[0]))
            except Exception as e:
                print(f"Error loading or processing {image_path}: {e}")
        return images, labels

    # Load images and label from dataset folder
    images, labels = get_images_and_labels(dataset_path)
    print("Number of images:", len(images))
    print("Number of labels:", len(set(labels)))

    if len(images) == 0:
        print("No face images found in the dataset folder!")
        exit()

    # Perform manual cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(images):
        X_train, X_test = np.array(images)[train_index], np.array(images)[test_index]
        y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]

        # Train the recognizer using the training data
        recognizer.train(X_train, np.array(y_train))

        # Make predictions on the testing data
        y_pred = []
        for image in X_test:
            label, confidence = recognizer.predict(image)
            y_pred.append(label)

        # Calculate accuracy score
        score = accuracy_score(y_test, y_pred)
        scores.append(score)

    # Print cross-validation scores
    print("Cross-Validation Scores:", scores)
    print("Mean Accuracy:", np.mean(scores))
    print("Standard Deviation:", np.std(scores))

    # Train the recognizer on the entire dataset
    recognizer.train(images, np.array(labels))

    # Save the recognizer to a file
    recognizer.save("lbph_modelFYPREAL1.yml")
    messagebox.showinfo("Update Model","Model Has Been Update")
#_________________________________________________________________________________________________________________________________________________________

#_________________________________________________________________________________________________________________________________________________________
#Face Real Time
def real():
    # Load the LBPH model
    model_path = "lbph_modelFYPREAL1.yml"
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(model_path)

    # Load the cascade classifier
    cascade_path = "fyp\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Set up the video capture object using the webcam (index 0)
    cap = cv2.VideoCapture(1)

    # Define a dictionary that maps label numbers to names
    label_name_dict = {
        1: "DrNur Huda",

    }

    # Set to keep track of recorded names
    recorded_names = set()

    # Function to write the attendance data to the CSV file
    def write_attendance_to_csv(label_name, timestamp):
        with open('attendance.csv', mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([label_name, timestamp])

    # Loop to capture frames from the webcam and perform face recognition
    while True:
        try:
            # Read a frame from the webcam
            ret, frame = cap.read()

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = face_cascade.detectMultiScale(gray)

            # Predict the label and confidence score for each detected face
            for (x, y, w, h) in faces:
                # Extract the face region of interest (ROI) from the grayscale frame
                roi_gray = gray[y:y + h, x:x + w]

                # Predict the label and confidence score for the ROI
                label, confidence = model.predict(roi_gray)

                if label in label_name_dict:
                    # Get the name corresponding to the predicted label from the label_name_dict
                    name = label_name_dict[label]
                else:
                    name = "UNKNOWN!!!"

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Add the predicted name and confidence score as text
                text = f"{name}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Add the detected name to the attendance list if it's not already there
                if name != "UNKNOWN!!!" and name not in recorded_names:
                    # Get the current timestamp
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # Write the label and timestamp to the CSV file
                    write_attendance_to_csv(name, timestamp)
                    # Add the name to the recorded_names set to avoid duplicates
                    recorded_names.add(name)

            # Display the frame
            cv2.imshow("Face Recognition", frame)

            # Exit the loop and release the video capture object when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error as e:
            print(f"OpenCV error: {e}")
            break

    # Release the video capture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo(Attendance_icon_image,"Attendance Has Been Capture")


#_________________________________________________________________________________________________________________________________________________________




#_________________________________________________________________________________________________________________________________________________________
#Button for sign in staff
def SignInStaff_pressed():
    # Get the entered username
    username = user_name_input.get()

    # Remove all other widgets from the main page
    user_name_label.place_forget()
    user_password_label.place_forget()
    submit_button.place_forget()
    user_name_input_area.place_forget()
    user_password_input.place_forget()
    checkbox.place_forget()
    MainPage.place_forget()
    MainPage1.place_forget()
    mainpage2.place_forget()
    submit_button2.place_forget()

    message_label = Label(top, text="", bg="black", fg="white")

    # Clear the message label
    message_label.config(text="")

    # Update the message label with the personalized message
    message_label.config(text=f"Welcome {username}! to UiTM FaceTrack!!!.", font=("Times New Roman",13), bg="white", fg="#4169e1")

    # Create a new button that pops up after submitting
    UpdataDataset_button.place(x=10, y=230) 
    Updated_button.place(x=10, y=340)
    Database_button.place(x=10, y=420)   #BUTTON AUGUST 
    message_label.place(x=75, y=180)      
    Database1_button.place(x=10, y=450)  #BUTTON SEPTEMBER
    LogOut_button.place(x=200, y=460)    #LOG OUT BUTTON

    # Display the additional input field
    StudentID_input.place(x=100, y=265)
    StudentName_input.place(x=100, y=295)

    #create a short description text
    DescDataset = Label(top, text=" Notes* makesure your camera is functional", font=("Times New Roman",13),bg= "white", fg="#000080")
    DescDataset.place(x=5, y=210)
    DescStudentID = Label(top,text="Student ID       :", font=("Times New Roman",10),bg= "white", fg="black")
    DescStudentID.place(x=10, y=265)
    DescStName = Label(top,text="Student Name :", font=("Times New Roman",10),bg= "white", fg="black")
    DescStName.place(x=10, y=295)
    UpdateModel = Label(top, text="REMINDER: Keep The Model Update For Every Semester", font=("Times New Roman",10),bg= "white", fg="#000080")
    UpdateModel.place(x=105, y=340)
    DatabaseSpace = Label(top,text="=========================================================", font=("Times New Roman",10),bg= "white", fg="blue")
    DatabaseSpace.place(x=10, y=380)
    DatabaseInfo = Label(top,text="Student CS259 6C Attendance Database 2023", font=("Times New Roman",10),bg= "white", fg="#000080")
    DatabaseInfo.place(x=10, y=400)
#_________________________________________________________________________________________________________________________________________________________


#_________________________________________________________________________________________________________________________________________________________
#open csv file
def Database():
    file_path = r"C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\attendance.csv"
    
    try:
        with open(file_path, 'r', newline='') as file:
            pass  # Do nothing and simply open the file

        # Open the file using the default application associated with CSV files
        subprocess.Popen(["start", "", file_path], shell=True)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
#_________________________________________________________________________________________________________________________________________________________


#_________________________________________________________________________________________________________________________________________________________
#Log out button
def LogOut():
    top.destroy()
#_________________________________________________________________________________________________________________________________________________________



#_________________________________________________________________________________________________________________________________________________________
#SignIn model    
def SignIn():

    #design template 
    top.configure(bg="#009feb")
    tajuk = Label(top, text="Face Track - Hidup UITM", font=("Roboto",24),bg="#009feb", fg="white")
    tajuk.place(x=50, y =130)

    #Remove all other widgets from the main page
    user_name_label.place_forget()
    user_password_label.place_forget()
    submit_button.place_forget()
    user_name_input_area.place_forget()
    user_password_input.place_forget()
    checkbox.place_forget()
    MainPage.place_forget()
    MainPage1.place_forget()
    mainpage2.place_forget()
    submit_button2.place_forget()
    
    # Get the entered username
    username = user_name_input.get()
    message_label = Label(top, text="", bg="black", fg="white")

    # Clear the message label
    message_label.config(text="")
    message_label.place(x=75, y=180)  

    # Update the message label with the personalized message
    message_label.config(text=f"Welcome {username}! to UiTM FaceTrack!!!.", font=("Times New Roman",15), bg="#009feb", fg="white")

    #create a new button
    Attendance_button.place(x=120, y=220)
    TakeAttendance_button.place(x=160, y=420)
    LogOut1_button.place(x=200, y=460)    #LOG OUT BUTTON
#_________________________________________________________________________________________________________________________________________________________


#_________________________________________________________________________________________________________________________________________________________ 
#open camera take dataset
import os
import cv2
import tkinter as tk
from tkinter import filedialog

def camera():
    # Load the Haar Cascade face detector
    cascade_path = "fyp\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Start capturing video from the default camera (index 0)
    cap = cv2.VideoCapture(0)

    count = 0

    # Create "DatasetPRESENT DAY" folder if it doesn't exist
    folder_path = "datasetBase"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    while True:
        # Capture each frame from the camera
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over the detected faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Update GUI id and name (assuming you have a top and StudentID_input, StudentName_input defined elsewhere)
            top.update_idletasks()
            DescStudentID_str = StudentID_input.get()
            DescStudentName_str = StudentName_input.get()

            # Save image in the "DatasetPRESENT DAY" folder
            image_name = os.path.join(folder_path, f"{DescStudentID_str}_{DescStudentName_str}_{count}.png")
            cv2.imwrite(image_name, frame[y:y+h, x:x+w])

        # Display the frame with face detections
        cv2.imshow('Face Detection', frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Set count
        elif count >= 500:
            break

    # Release the video capture and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Update Dataset","Dataset Has Been Taken")
#_________________________________________________________________________________________________________________________________________________________


#starting of GUI
#DESIGN PAGE
top = Tk()
top.geometry("450x500")
top.configure(bg="white")
top.title("Face - Track")
#add image UITM
bg = PhotoImage( file = "logouitm.png" )
label1 = Label(top, image=bg)
label1.place(x=0, y=0)


MainPage = Label(top, text="Welcome Back !", font=("Times New Roman",13),bg= "white", fg="#4169e1")
MainPage.place(x=15, y=180)
MainPage1 = Label(top, text="Sign In to continue to FaceTrack Portal", font=("Times New Roman",10),bg= "white", fg="#4169e1")
MainPage1.place(x=15, y=200)
tajuk = Label(top, text="Face Track - Hidup UITM", font=("Roboto",24),bg="white", fg="purple")
tajuk.place(x=50, y =130)
mainpage2 = Label(top,text="OR", font=("Times New Roman",13), bg="white", fg="grey" )
mainpage2.place(x= 150 , y=390) 

#MAIN PAGE
# the label for user_name and user_password and submit button)
user_name_label = Label(top, text="Username", bg="white")
user_password_label = Label(top, text="Staff ID", bg="white")
submit_button = Button(top, text="Sign-In", width=42, bg="#4169e1", fg="white" ,command=SignIn)
var1 = IntVar()
checkbox = Checkbutton(top, text="Remember me", variable=var1, bg="white")
submit_button2 = Button(top, text="Sign-In Staff", width=42, bg="#191970", fg="white" ,command=SignInStaff_pressed)



# Create a StringVar to store the entered username
user_name_input = StringVar()
user_name_input_area = Entry(top, width=50, textvariable=user_name_input,bg="#eaf2fa")

# Create an Entry widget to capture the password (you can set the 'show' option to '*' for password masking)
user_password_input = Entry(top, width=50, bg="#eaf2fa" )


# Create the new button (your name button) and input in function sign in staff
UpdataDataset_button = Button(top, text="Update Dataset", bg="#20bebe", command=camera)
StudentID_input = Entry(top, width=30, bg="#eaf2fa")
Updated_button = Button (top, text="Update Model", bg="#20bebe", command=train)
StudentName_input = Entry(top, width=30, bg="#eaf2fa")
Database_button = Button (top, width=11, text="August", bg="#20bebe", command=Database)
Database1_button = Button(top, width=11, text="September", bg="#20bebe")
LogOut_image = PhotoImage(file="th.png")
resized_icon_image = LogOut_image.subsample(7, 7)
# Create the button with the resized house icon
LogOut_button = Button(top, image=resized_icon_image, command=LogOut) #log out page
#LogOut_button.pack()
Attendance_image = PhotoImage(file="face.png")
Attendance_icon_image = Attendance_image.subsample(1, 1)
Attendance_button = Button(top, image=Attendance_icon_image, command=real)
TakeAttendance_button = Button (top, text="Take Attendance", bg="white", fg="#20bebe" , command=open_video)
LogOut1_image = PhotoImage(file="log.png")
resized_log_image = LogOut1_image.subsample(7, 7)
LogOut1_button = Button(top, image=resized_log_image, command=LogOut) #log out page





# Place the widgets initially on the main page
user_name_label.place(x=15, y=220)
user_password_label.place(x=15, y=280)
submit_button.place(x=15, y=360)
user_name_input_area.place(x=15, y=250)
user_password_input.place(x=15, y=310)
checkbox.place(x=15, y=330)
submit_button2.place(x=15, y=420)


top.mainloop()
