import os
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

count = 0

while(True):
    ret, raw_bgr = cap.read()
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    bgr_inverted = cv2.threshold(raw_bgr, 50, 255, cv2.THRESH_BINARY_INV)
    resized_bgr = cv2.resize(bgr_inverted[1], (28, 28), interpolation = cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    rgb_tensor = (tf.convert_to_tensor(rgb, dtype=tf.float32) / 255.0)
    gray_tensor = tf.image.rgb_to_grayscale(rgb_tensor)
    cv2.imwrite("preview.jpg", gray_tensor.numpy())
    gray_tensor = tf.expand_dims(gray_tensor, axis=0)
    y_hat = loaded_model.predict(gray_tensor)

    prediction= np.argmax(y_hat)
    score = y_hat[[0],[prediction]].item()
    if score >= 0.7:
        print('Number is a {} with a certainty of {:.2%}'.format(prediction, score))
        sense.show_letter(str(prediction))
    else:
        if count % 2:
            sense.show_letter('/')
        else:
            sense.show_letter('\\')
        count = count + 1

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
   
        
cap.release()
cv2.destroyAllWindows()