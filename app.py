# Importing the Libraries
#tkinter module for GUI development
import tkinter as tk
from tkinter import filedialog, messagebox
# PIL library for image processing
from PIL import ImageTk, Image 
# numpy for numerical operations
import numpy as np
# TensorF for deep learning tasks
import tensorflow as tf
from keras.preprocessing import image
#custom class names from a module
from resources.class_names import class_1, class_2

# Load pre-trained models
image_classifier_model = tf.keras.models.load_model('./notebooks/models/image_classifier_model.keras')
pneumonia_detection_model = tf.keras.models.load_model('./notebooks/models/pneumonia_detection_model.keras')

# image dimensions
IMAGE_SIZE = (180, 180)


# Loading class names
image_classification_names = class_1
pneumonia_detection_names = class_2

# Function to handle selection of functionalities
def select_function(selection):
    if selection == 1:
        image_classification()
    elif selection == 2:
        pneumonia_detection()

# Function to upload image for image classification
def upload_image_classification():
    filepath = filedialog.askopenfilename(initialdir="./test_cases/inputs/Image_classification/")
    if filepath:
        if check_image_file(filepath):
            display_image_classification(filepath)
        else:
            messagebox.showerror("Error", "Invalid file format. Please select an image file.")

# Function to check if a file is a valid image file
def check_image_file(filepath):
    try:
        with Image.open(filepath):
            return True
    except:
        return False

# Function to open window for image classification
def image_classification():
    window01 = tk.Toplevel(root) 
    window01.title("Image Classification")
    window01.geometry("400x200")  
    window01.configure(bg="black")

    bg_image = ImageTk.PhotoImage(Image.open("./resources/images/image_classification_bg.jpg").resize((400, 200)))
    bg_label = tk.Label(window01, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    upload_button = tk.Button(window01, text="Upload Image", command=upload_image_classification,
                              bg="black", fg="yellow", highlightbackground="black", borderwidth=3,
                              font=("Times New Roman", 12))
    upload_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    exit_button = tk.Button(window01, text="Exit", command=window01.destroy,
                            bg="black", fg="yellow", highlightbackground="black", borderwidth=2,
                            font=("Times New Roman", 10))
    exit_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    window01.mainloop()

# Function to display selected image for image classification
def display_image_classification(filepath):
    image = Image.open(filepath)
    image = image.resize((300, 300))
    photo = ImageTk.PhotoImage(image)
    
    window11 = tk.Toplevel(root)
    window11.title("Image Display")
    window11.geometry("500x400")
    window11.configure(bg="black")
    
    image_label = tk.Label(window11, image=photo)
    image_label.image = photo
    image_label.pack()
    
    predict_image(filepath, window11)

# Function to predict class of image
def predict_image(filepath, window):
    img_load = tf.keras.utils.load_img(filepath, target_size=IMAGE_SIZE)
    img_arr = tf.keras.utils.img_to_array(img_load)
    img_batch = np.expand_dims(img_arr, 0)
    
    predict = image_classifier_model.predict(img_batch)
    score = tf.nn.softmax(predict)
    class_name = image_classification_names[np.argmax(score)].split(".")
    class_text = tk.Label(window, text=f"Predicted image as: {class_name[1]}",
                          font=("Times New roman", 13, "bold"), fg="yellow",  bg="black")
    class_text.pack(pady=10)
    
    exit_button = tk.Button(window, text="Exit", command=window.destroy,
                            bg="black", fg="yellow", highlightbackground="black",
                            borderwidth=2, font=("Times New Roman", 10))
    exit_button.pack()

# Function to upload a image for pneumonia detection
def upload_pneumonia_detection():
    filepath = filedialog.askopenfilename(initialdir="./test_cases/inputs/pneumonia_detection/")
    if filepath:
        if check_image_file(filepath):
            display_pneumonia_detection(filepath)
        else:
            messagebox.showerror("Error", "Invalid file format. Please select an image file.")

# Function to open window for pneumonia detection
def pneumonia_detection():
    window02 = tk.Toplevel(root)
    window02.title("Pneumonia Detection")
    window02.geometry("400x200")
    window02.configure(bg="black")

    bg_image = ImageTk.PhotoImage(Image.open("./resources/images/pneumonia_detection_bg.jpg").resize((400, 200)))
    bg_label = tk.Label(window02, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    upload_button = tk.Button(window02, text="Upload Image", command=upload_pneumonia_detection,
                              bg="black", fg="yellow", highlightbackground="black", borderwidth=3,
                              font=("Times New Roman", 12))
    upload_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    exit_button = tk.Button(window02, text="Exit", command=window02.destroy,
                            bg="black", fg="yellow", highlightbackground="black", borderwidth=2,
                            font=("Times New Roman", 10))
    exit_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    window02.mainloop()

# Function to display selected image for pneumonia detection
def display_pneumonia_detection(filepath):
    image = Image.open(filepath)
    image = image.resize((300, 300))
    photo = ImageTk.PhotoImage(image)
    
    window22 = tk.Toplevel(root)
    window22.title("Image Display")
    window22.geometry("500x400")
    window22.configure(bg="black")
    
    image_label = tk.Label(window22, image=photo)
    image_label.image = photo
    image_label.pack()
    
    predict_pneumonia(filepath, window22)

# Function to predict the pneumonia
def predict_pneumonia(filepath, window):
    img_load = image.load_img(filepath, target_size=IMAGE_SIZE)
    img_arr = image.img_to_array(img_load)
    img_batch = np.expand_dims(img_arr, axis=0)
    
    predict = pneumonia_detection_model.predict(img_batch)
    class_name = pneumonia_detection_names[int(np.round(predict[0][0]))]
    class_text = tk.Label(window, text=f"Predicted as: {class_name}",
                          font=("Times New roman", 13, "bold"), fg="yellow",  bg="black")
    class_text.pack(pady=10)
    
    exit_button = tk.Button(window, text="Exit", command=window.destroy,
                            bg="black", fg="yellow", highlightbackground="black",
                            borderwidth=2, font=("Times New Roman", 10))
    exit_button.pack()

# Creating main root window for GUI
root = tk.Tk()
root.title("Image Recognition System")
root.geometry("400x200")

background_image = ImageTk.PhotoImage(Image.open("./resources/images/root_bg.jpg").resize((400, 200)))
canvas = tk.Canvas(root, width=400, height=200)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, anchor=tk.NW, image=background_image)

var = tk.IntVar()
radio_button1 = tk.Radiobutton(root, text="Image Classification", variable=var, value=1)
radio_button1.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

radio_button2 = tk.Radiobutton(root, text="Pneumonia Detection", variable=var, value=2)
radio_button2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

select_button = tk.Button(root, text="Select", command=lambda: select_function(var.get()),bg="black", fg="yellow", 
                          highlightbackground="black", borderwidth=2,  font=("Times New Roman", 12))
select_button.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

exit_button = tk.Button(root, text="Exit", command=root.destroy, bg="black", fg="yellow",
                        highlightbackground="black", borderwidth=2,  font=("Times New Roman", 10))
exit_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

root.mainloop()
