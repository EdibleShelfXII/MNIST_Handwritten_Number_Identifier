import os
import time
import keyboard
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from sense_hat import SenseHat

print(tf.version.VERSION)
sense = SenseHat()


loaded_model = tf.keras.models.load_model('/home/pi/handrecog/keras_convnet_adam')

loaded_model.summary()

cap = cv2.VideoCapture(0)

#sense.show_message("Start")

# define the contrast and brightness value
contrast = 0. # Contrast control ( 0 to 127)
brightness = 0. # Brightness control (0-100)

while(True):
    ret, raw_bgr = cap.read()
    #cv2.normalize(frame, frame, 20, 235, cv2.NORM_MINMAX)
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    #cv2.imwrite("img.jpg", frame)
    #raw_bgr = cv2.imread("img.jpg")
    bgr_inverted = cv2.threshold(raw_bgr, 50, 255, cv2.THRESH_BINARY_INV)
    #raw_contrast_adj = cv2.addWeighted( raw_bgr, contrast, raw_bgr, 0, brightness)
    resized_bgr = cv2.resize(bgr_inverted[1], (28, 28), interpolation = cv2.INTER_CUBIC)
    #cv2.imshow('processed image', img)
    #print(img.shape)
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
    gray_tensor = tf.image.rgb_to_grayscale(rgb_tensor)
    #print(img.shape)
    cv2.imwrite("preview.jpg", gray_tensor.numpy())
    gray_tensor = tf.expand_dims(gray_tensor, axis=0)
    #print(img.shape)
    y_hat = loaded_model.predict(gray_tensor)
    print(f"{y_hat}")
    prediction= np.argmax(y_hat)
    score = y_hat[[0],[prediction]].item()
    print(score)
    print('Number is a {} with a certainty of {:.2%}'.format(prediction, score))
    sense.show_letter(str(prediction))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
   
        
cap.release()
cv2.destroyAllWindows()