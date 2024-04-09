import os
from pathlib import Path
import pickle
import numpy as np
import face_recognition
from PyQt5.QtCore import QRunnable, pyqtSignal


class Face:
    def __init__(self, location: list[int], encoding: np.ndarray, tmp_name=None, name=None, auto=False) -> None:
        self.location = location
        self.encoding = encoding
        self.tmp_name = tmp_name
        self.name = name
        self.auto = auto # if the face was named automatically


class Image:
    def __init__(self, image_path: str | Path, locations: list, encodings: list, threshold=None) -> None:
        self.faces = []
        self.image_path = image_path
        self.threshold = threshold # the one used to classify this image
        x_coors = np.array([l[1] for l in locations])
        sort_idxs = np.argsort(x_coors)
        for idx in sort_idxs:
            face = Face(locations[idx], encodings[idx])
            self.faces.append(face)

class FaceDetector(QRunnable):
    def __init__(self, image_path:Path, encodings_folder:Path):
        self.image_path = image_path
        self.encodings_folder= encodings_folder
        super().__init__()
        
    def run(self):
        if not self.image_path.is_file():
            print("wtf")
            return
        print(f"Detecting faces: {self.image_path.name}")
        out_path = os.path.join(self.encodings_folder, self.image_path.stem+".pickle")
        if os.path.exists(out_path):
            print("wtf")
            return

        image = face_recognition.load_image_file(self.image_path, mode='RGB')
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(
            image, 
            known_face_locations=locations
            )
        image = Image(self.image_path, locations, encodings)

        with open(out_path, 'wb') as f:
            pickle.dump(image, f)
        print("ok")

class NotThreadedFaceDetector:
    def __init__(self, image_path:Path, encodings_folder:Path):
        self.image_path = image_path
        self.encodings_folder= encodings_folder
        
    def run(self):
        if not self.image_path.is_file():
            return
        print(f"Detecting faces: {self.image_path.name}")
        out_path = os.path.join(self.encodings_folder, self.image_path.stem+".pickle")
        if os.path.exists(out_path):
            return

        image = face_recognition.load_image_file(self.image_path, mode='RGB')
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(
            image, 
            known_face_locations=locations
            )
        image = Image(self.image_path, locations, encodings)

        with open(out_path, 'wb') as f:
            pickle.dump(image, f)

        