from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import pickle
import numpy as np
import face_recognition
from PyQt5.QtCore import QObject

import logging
logger = logging.getLogger(__name__)

class Face:
    def __init__(self, location: list[int], encoding: np.ndarray, name=None, auto=False) -> None:
        self.location = location
        self.encoding = encoding
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

class FaceDetector:
    # Static for performance reasons
    @staticmethod
    def run(args):
        image_path, encodings_folder = args
        if not image_path.is_file():
            return
        print(f"Detecting faces: {image_path.name}")
        out_path = os.path.join(encodings_folder, image_path.stem+".pickle")
        if os.path.exists(out_path):
            return

        image = face_recognition.load_image_file(image_path, mode='RGB')
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(
            image, 
            known_face_locations=locations
            )
        image = Image(image_path, locations, encodings)

        with open(out_path, 'wb') as f:
            pickle.dump(image, f)
    
class MuliprocessFaceDetector(QObject):
    def __init__(self, args):
        super().__init__()
        self.args = args
        num_cores = cpu_count()
        self.pool = Pool(num_cores*2)
        logger.debug("Please look at the console to see the progress")

    def run(self):
        self.pool.map(FaceDetector.run, self.args)
        self.pool.close()
        self.pool.join()
        logger.debug("Done")

    def deleteLater(self) -> None:
        self.pool.terminate()
        super().deleteLater()