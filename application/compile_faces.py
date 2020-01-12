import json
import face_recognition
import pickle

all_face_encodings = {}

try:
    faces = open("faces/people.json")
except Exception as err:
    print("Could not open file: %s", err)
    raise

json_data = json.load(faces)
people = json_data["People"]

for face in people:
    img = face_recognition.load_image_file("faces/" + face["image"])
    all_face_encodings[face["name"]] = face_recognition.face_encodings(img)[0]

with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)