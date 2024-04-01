from PyQt5.QtCore import QStringListModel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCompleter
import sys
import pickle
import face_recognition
from pathlib import Path
import argparse
import cv2
import os
from multiprocessing import Pool, cpu_count
from PyQt5.QtCore import QRect

import numpy as np

out_folder = Path("detections")

process_number = cpu_count()
threshold = 0.66


class Face:
    def __init__(self, location: list[int], encoding: np.ndarray, tmp_name=None, name=None) -> None:
        self.location = location
        self.encoding = encoding
        self.tmp_name = tmp_name
        self.name = name


class ImageFaces:
    def __init__(self, image_path: str | Path, locations: list, encodings: list) -> None:
        self.faces = []
        self.image_path = image_path
        x_coors = np.array([l[1] for l in locations])
        sort_idxs = np.argsort(x_coors)
        for idx in sort_idxs:
            face = Face(locations[idx], encodings[idx])
            self.faces.append(face)


def detect_faces(image_paths):
    pool = Pool(process_number)
    pool.map(detect_face, image_paths)
    pool.close()
    pool.join()


def compare_faces(encoded_img_paths):
    unclassified_images = []
    pickle_paths = []
    for encoded_img_path in encoded_img_paths:
        with open(encoded_img_path, 'rb') as f:
            unclassified_images.append(pickle.load(f))
            pickle_paths.append(encoded_img_path)

    print(f"iterating through the {len(unclassified_images)} images")
    known_faces = []
    while len(unclassified_images) > 0:
        image_faces = unclassified_images.pop()
        pickle_path = pickle_paths.pop()
        image = cv2.imread(str(image_faces.image_path))
        print(f"iterating on {image_faces.image_path}")
        print(len(image_faces.faces))
        for face in image_faces.faces:
            if not known_faces:
                face.tmp_name = 0
                known_faces.append(face.encoding)
                continue
            distances = face_recognition.face_distance(
                known_faces, face.encoding)
            matcher = np.argmin(distances)
            print(distances.shape, distances[matcher])
            if distances[matcher] < threshold:
                face.tmp_name = matcher
            else:
                face.tmp_name = len(known_faces)
            known_faces.append(face.encoding)
            location = face.location
            start_point, end_point = (
                location[1], location[0]), (location[3], location[2])
            cv2.rectangle(image, start_point, end_point, 100)
            cv2.putText(image, str(face.tmp_name), start_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6)

        # save back the image_faces with the names
        with open(pickle_path, 'wb') as f:
            # pickle.dump(image_faces, f)
            pass

        cv2.imwrite(os.path.join(
            out_folder, image_faces.image_path.name), image)


def detect_face(image_path):
    print(f"detecting on {image_path}")
    out_path = os.path.join(out_folder, image_path.stem+".pickle")
    if os.path.exists(out_path):
        return

    image = face_recognition.load_image_file(image_path, mode='RGB')
    locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(
        image, known_face_locations=locations)
    image_faces = ImageFaces(image_path, locations, encodings)

    with open(out_path, 'wb') as f:
        pickle.dump(image_faces, f)


detections_folder = out_folder


class NameSelector(QWidget):
    def __init__(self, classified_folder, encoded_img_paths):
        super().__init__()
        self.classified_folder = classified_folder
        self.encoded_img_paths = encoded_img_paths
        self.next_image = True
        self.known_faces_encoding = []
        self.known_faces_name = [] # there can be repetitions, it is the same order as the encoding
        self.pickle_path = None
        self.propositions, self.distances = [], []

        # Set up GUI components
        self.image_label = QLabel()
        self.input_label = QLabel("Enter name or number:")
        self.input_field = Editor()
        self.submit_button = QPushButton("Submit")

        self.propositions_label = QLabel()
        self.submit_button.clicked.connect(self.handle_name)
        completer = QCompleter()
        completer.setCaseSensitivity(Qt.CaseInsensitive)

        self.input_field.setCompleter(completer)

        # Layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.propositions_label)
        self.layout.addWidget(self.input_label)
        self.layout.addWidget(self.input_field)
        self.layout.addWidget(self.submit_button)
        self.setLayout(self.layout)
        self.input_field.setFocus()
        self.next()

        # Set focus to input field


    def handle_action(self, action):
        if action <= 3:
            self.save_face(self.propositions[action-1])
        elif action == 4:
            # Unknown
            pass
        elif action == 5:
            # Skip all the faces in the image
            self.next_image = True
        elif action == 6:
            # Not a face
            self.image_faces.faces.remove(self.face)
        self.input_field.clear()
        self.next()

    def handle_name(self):
        user_input = self.input_field.text()
        if user_input.isdigit():
            return self.handle_action(int(user_input))
        else:
            user_input = ' '.join(word.capitalize()
                                    for word in user_input.split())
            self.save_face(user_input)
        self.input_field.clear()
        self.next()

    def save_face(self, name):
        self.face.name = name
        self.known_faces_name.append(name)
        self.known_faces_encoding.append(self.face.encoding)

        # create dir if not exist
        folder = os.path.join(self.classified_folder, name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        # create a simlink to the image
        path = os.path.join(folder, self.image_faces.image_path.name)
        if not os.path.islink(path):
            os.symlink(self.image_faces.image_path, path)

    def next(self):
        if self.next_image:
            if self.pickle_path:
                # save back the image_faces with the names
                with open(self.pickle_path, 'wb') as f:
                    pickle.dump(self.image_faces, f)

            try: 
                self.pickle_path = next(self.encoded_img_paths)
            except StopIteration:
                print("no more images")
                exit(0)

            try: 
                with open(self.pickle_path, 'rb') as f:
                    self.image_faces = pickle.load(f)
            except:
                print(f"error loading {self.pickle_path}")
                print("skipping")
                self.pickle_path = None
                self.next()
                
            self.faces = iter(self.image_faces.faces)
            self.next_image = False

        try:
            self.face = next(self.faces)
        except StopIteration:
            self.next_image = True
            self.next()
            return

        if self.face.name:
            print(f"known photo {self.image_faces.image_path} containing {self.face.name}")
            self.save_face(self.face.name)
            self.next()
            return
        imgdata = open(self.image_faces.image_path, 'rb').read()
        image = QImage.fromData(imgdata, self.image_faces.image_path.suffix)
        top, right, bottom, left = self.face.location
        width, heigth = right - left, bottom - top
        rect = QRect(left, top, width, heigth)
        image = image.copy(rect)
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.make_propositions()

    def make_propositions(self):
        if not self.known_faces_encoding:
            self.render_propositions()
            return
        self.distances = face_recognition.face_distance(
            self.known_faces_encoding,
            self.face.encoding
        )
        # reverse sort and select the 3 lowest distances
        # TODO: optimize
        matcher = -np.argsort(-self.distances) # reverse sort
        proposer = matcher[:3]
        self.distances = self.distances[proposer]
        if self.distances[0] < threshold:
            name = self.known_faces_name[proposer[0]]
            print(f"skipped photo {self.image_faces.image_path} containing {name} with distance {self.distances[0]}")
            self.save_face(name)
        else:
            self.propositions = [
                self.known_faces_name[i] for i in proposer
            ]
            self.render_propositions()

    def render_propositions(self):
        propositions_lines = [f"{i+1} : {prop} ({score})" for i,
                        (prop, score) in enumerate(zip(self.propositions, self.distances))]
        propositions_lines.append("4 : Unknown")
        propositions_lines.append("5 : Unknown for all faces in the image")
        propositions_lines.append("6 : Not a face")
        self.propositions_label.setText(
            "\n".join(propositions_lines)
        )


class Editor(QLineEdit):
    def __init__(self):
        super().__init__()

    def clear(self):
        super().clear()
        unique_known_faces_name = set(self.parent().known_faces_name)
        self.completer().setModel(QStringListModel(unique_known_faces_name))

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        text = event.text()
        if text.isdigit():
            self.parent().handle_action(int(text))
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.parent().handle_name()


def run_name_selector(classified_folder, encoded_img_paths):
    app = QApplication(sys.argv)
    window = NameSelector(classified_folder, encoded_img_paths)
    window.show()
    sys.exit(app.exec_())


def main():
    parser = argparse.ArgumentParser(
        description='Detect Faces')
    parser.add_argument(
        '-f', '--folder', help='Picture folders', type=str, default='demo')
    parser.add_argument(
        '-c', '--compare', help='Compare the created encoding only', action='store_true')
    parser.add_argument(
        '-r', '--restart', help='Restart the classification', action='store_true')
    args = parser.parse_args()
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    folder = Path(args.folder)
    classified_folder =    folder.parent/Path("classified")
    image_paths = folder.rglob("*")
    if args.restart:
        # select the files of every subfolder and remove them using pathlib
        for file in classified_folder.glob("*/*"):
            os.remove(file)
    if not args.compare:
        detect_faces(image_paths)

    encoded_img_paths = Path(out_folder).rglob("*.pickle")
    run_name_selector(classified_folder, encoded_img_paths)


if __name__ == "__main__":
    main()

# python.exe .\script.py --folder="C:\Users\bcbav\Downloads\print_photos"
