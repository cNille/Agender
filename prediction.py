import cv2
import os
import imutils
import time
import numpy as np
from matplotlib import pyplot as plt

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = [
    '(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)',
    '(48, 53)', '(60, 100)'
]
gender_list = ['Male', 'Female']


class Agender:
    def __init__(self):

        self.age_net = cv2.dnn.readNetFromCaffe('./data/deploy_age.prototxt',
                                                './data/age.caffemodel')

        self.gender_net = cv2.dnn.readNetFromCaffe(
            './data/deploy_gender.prototxt', './data/gender.caffemodel')
        self.face_cascade = cv2.CascadeClassifier(
            'data/haarcascade_frontalface_alt.xml')
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def predict(self, image_path):
        print('Predicting %s' % image_path)

        # Load image
        image = cv2.imread(image_path)

        # Scale to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        print("Found {} faces".format(str(len(faces))))

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Get Face
            face_img = image[y:y + h, h:h + w].copy()
            blob = cv2.dnn.blobFromImage(
                face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            #Predict Age
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print("Age Range: " + age)

            #Predict Gender
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            print("Gender : " + gender)

            overlay_text = "%s %s" % (gender, age)
            cv2.putText(image, overlay_text, (x, y), self.font, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', image)
        cv2.waitKey(3000)  #pauses for 3 seconds
