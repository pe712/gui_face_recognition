from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import pickle
import numpy as np
import face_recognition


class Face:
    def __init__(self, location: list[int], encoding: np.ndarray, tmp_name=None, name=None, auto=False) -> None:
        self.location = location
        self.encoding = encoding
        self.tmp_name = tmp_name
        self.name = name
        self.auto = auto # if the face was named automatically


class Image:
    def __init__(self, image_path: str | Path, locations: list, encodings: list) -> None:
        self.faces = []
        self.image_path = image_path
        x_coors = np.array([l[1] for l in locations])
        sort_idxs = np.argsort(x_coors)
        for idx in sort_idxs:
            face = Face(locations[idx], encodings[idx])
            self.faces.append(face)

class FaceDetector:
    def __init__(self, image_folder, out_folder):
        self.image_folder = image_folder
        self.out_folder = out_folder
        self.image_paths = list(self.image_folder.rglob("*"))
  
    def detect_faces(self):
        process_number = cpu_count()

        pool = Pool(process_number)
        pool.map(FaceDetector.detect_face, zip(self.image_paths, [self.out_folder]*len(self.image_paths)))
        pool.close()
        pool.join()
    
    @staticmethod
    def detect_face(arg):
        """
        For performance reasons, it is important that this method is static
        """
        image_path, out_folder = arg
        if not image_path.is_file():
            return
        print(f"Detecting faces: {image_path.name}")
        out_path = os.path.join(out_folder, image_path.stem+".pickle")
        if os.path.exists(out_path):
            return

        image = face_recognition.load_image_file(image_path, mode='RGB')
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(
            image, known_face_locations=locations)
        image = Image(image_path, locations, encodings)

        with open(out_path, 'wb') as f:
            pickle.dump(image, f)