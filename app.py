import joblib
import sklearn
import cv2
import os
import numpy as np
import random

from sklearn.preprocessing import LabelEncoder
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from ultralytics import YOLO




yolo_base_model = YOLO("yolov8n.pt")
objects_dict = {0:"person", 1:"bicycle", 2:"car", 3:"motorcycle", 4:"airplane", 5:"bus", 6:"train", 7:"truck", 8:"boat", 9:"traffic light", 10:"fire hydrant", 11:"stop sign", 12:"parking meter", 13:"bench", 14:"bird", 15:"cat", 16:"dog", 17:"horse", 18:"sheep", 19:"cow", 20:"elephant", 21:"bear", 22:"zebra", 23:"giraffe", 24:"backpack", 25:"umbrella", 26:"handbag", 27:"tie", 28:"suitcase", 29:"frisbee", 30:"skis", 31:"snowboard", 32:"sports ball", 33:"kite", 34:"baseball bat", 35:"baseball glove", 36:"skateboard", 37:"surfboard", 38:"tennis racket", 39:"bottle", 40:"wine glass", 41:"cup", 42:"fork", 43:"knife", 44:"spoon", 45:"bowl", 46:"banana", 47:"apple", 48:"sandwich", 49:"orange", 50:"broccoli", 51:"carrot", 52:"hot dog", 53:"pizza", 54:"donut", 55:"cake", 56:"chair", 57:"couch", 58:"potted plant", 59:"bed", 60:"dining table", 61:"toilet", 62:"tv", 63:"laptop", 64:"mouse", 65:"remote", 66:"keyboard", 67:"cell phone", 68:"microwave", 69:"oven", 70:"toaster", 71:"sink", 72:"refrigerator", 73:"book", 74:"clock", 75:"vase", 76:"scissors", 77:"teddy bear", 78:"hair drier", 79:"toothbrush"}




def load_image():
    global image
    global image_displayed
    
    # Open a box to choose a picture
    file_path = filedialog.askopenfilename()
    
    # Load the image
    image = cv2.imread(file_path)

    # display the image
    image_displayed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_displayed = Image.fromarray(image_displayed)
    image_displayed.thumbnail((400, 400))  #resizing the image
    image_tk = ImageTk.PhotoImage(image_displayed)
    image_label.configure(image=image_tk)
    image_label.image = image_tk

#prediction with the model trained on the dataset prepared without yolo
def predict_image():
    global image
    global image_pretreated 

    image_pretreated = pretreate_image(image)
    
    tableau_images = []
    tableau_images.append(image_pretreated)
    tableau_images = np.array(tableau_images)
    
    prediction = model.predict(tableau_images)
    
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=-1))

    # Prediction text
    text_label.set("Prédiction : " + predicted_label[0])

#prediction with the model trained on the dataset prepared with yolo
def predict_image_yolo():
    global image
    global image_pretreated 

    image_pretreated = pretreate_image(image)
    
    tableau_images = []
    tableau_images.append(image_pretreated)
    tableau_images = np.array(tableau_images)
    
    prediction = model.predict(tableau_images)
    
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=-1))

    
    # Prediction text
    text_label.set("Prédiction : " + predicted_label[0])

def resize_image_square(image, size):
    old_size = image.shape[:2]
    ratio = float(size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded_image

def pretreate_image(img):
    image = resize_image_square(img, 224)
    image = preprocess_input(image)
    return image

#0:person, 1:bicycle, 2:car, 3:motorcycle, 4:airplane, 5:bus, 6:train, 7:truck, 8:boat, 9:traffic light,
#10:fire hydrant, 11:stop sign, 12:parking meter, 13:bench, 14:bird, 15: cat, 16:dog, 17:horse, 18:sheep, 19:cow,
#20:elephant, 21:bear, 22:zebra, 23:giraffe, 24:backpack, 25:umbrella, 26:handbag, 27:tie, 28:suitcase, 29:frisbee,
#30:skis, 31:snowboard, 32:sports ball, 33:kite, 34:baseball bat, 35:baseball glove, 36:skateboard, 37:surfboard, 38:tennis racket, 39:bottle,
#40:wine glass, 41:cup, 42:fork, 43:knife, 44:spoon, 45:bowl, 46:banana, 47:apple, 48:sandwich, 49:orange,
#50:broccoli, 51:carrot, 52:hot dog, 53:pizza, 54:donut, 55:cake, 56:chair, 57:couch, 58:potted plant, 59:bed,
#60:dining table, 61:toilet, 62:tv, 63:laptop, 64:mouse, 65:remote, 66: keyboard, 67: cell phone, 68: microwave, 69: oven,
#70:toaster, 71:sink, 72:refrigerator, 73:book, 74:clock, 75:vase, 76:scissors, 77:teddy bear, 78:hair drier, 79:toothbrush

def crop_around_object(image, object_id=16):
  results = yolo_base_model.predict(image)
  for box in results[0].boxes.data:
    if int(box[-1]) == object_id:
      x_min, y_min, x_max, y_max = box[:4]
      cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
      return cropped_image, x_min, y_min, x_max-x_min, y_max-y_min
  print("Objet non trouvé")
  return image, 0, 0, 0, 0





image = None
image_pretreated = None
image_displayed = None
model = load_model('prediction_files/ResNet_model.h5')
label_encoder = joblib.load('prediction_files/label_encoder.joblib')
model_crop = load_model('prediction_files/ResNet_model_crop.h5')
label_encoder_crop = joblib.load('prediction_files/label_crop_encoder.joblib')



#User Interface
window = Tk()
window.geometry("800x600")
window.title("Prédiction de la race d'un chien")

file_button = Button(window, text="Sélectionner une image", command=load_image)
file_button.pack(pady=10)

frame = Frame(window)
frame.pack()

button1 = Button(frame, text="ResNet", command=predict_image)
button1.pack(padx=10)

button2 = Button(frame, text="ResNet & YOLO", command=predict_image_yolo)
button2.pack(padx=10)

text_label = StringVar()
prediction_label = Label(window, textvariable=text_label)
prediction_label.pack(pady=10)

image_label = Label(window)
image_label.pack()

window.mainloop()