import face_recognition as fr
import os
import cv2
import numpy as np

# Setting up variables

faces_folder = '../faces'
faces_test_folder = '../test'

known_face_encodings = []
known_face_names = []

# Reading know faces in folder

face_locations = []
known_face_encodings = []

for file in os.listdir(faces_folder):
    if file.endswith('.png'):
        im = fr.load_image_file(faces_folder+'/'+file)
        encoding = fr.face_encodings(im)
        known_face_encodings.append(encoding[0])
        face_name = file.split('.')[0].capitalize()
        known_face_names.append(face_name)
        print('Processing '+file+' found person '+face_name)        
