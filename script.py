from typing import Iterable
from PyQt5.QtCore import QStringListModel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCompleter
import sys
import pickle
import face_recognition
from pathlib import Path
import argparse
import os
from PyQt5.QtCore import QRect
import tempfile
import numpy as np

from face import Face, Image # needed for the pickle loading
from face import FaceDetector
from contacts import CSV

out_folder = Path("detections")
detections_folder = out_folder


UNKNOWN = "Unknown"
BAD_QUALITY = "Bad Quality"
AUTO = "Auto"
SKIPPED = "Skipped"
IMAGE_COUNT = "Image count"

class NameSelector(QWidget):
    def __init__(self, classified_folder, encoded_img_folder, contacts, name=None):
        super().__init__()
        self.classified_folder = classified_folder
        self.name = name
        
        # make a copy of the generator
        self.encoded_img_folder = encoded_img_folder
        self.encoded_img_paths = encoded_img_folder.glob("*")
        self.init_completions(encoded_img_folder.glob("*"), contacts)
        
        self.next_image = True
        self.known_faces_encoding = []
        self.known_faces_name = [] # there can be repetitions, it is the same order as the encoding
        self.pickle_path = None
        self.previous_pickle_path = None # the last action
        self.reverting = False
        self.propositions, self.distances = [], []

        self.threshold = 0.66
        
        self.stats = {BAD_QUALITY: 0, UNKNOWN: 0, AUTO: 0, SKIPPED: 0, IMAGE_COUNT: 0}

        self.k = 3

        # Set up GUI components
        self.image_label = QLabel()
        self.input_label = QLabel("Enter name or number:")
        self.input_field = Editor()
        self.submit_button = QPushButton("Submit")
        self.revert_button = QPushButton("Revert")

        self.propositions_label = QLabel()
        self.submit_button.clicked.connect(self.submit)
        self.revert_button.clicked.connect(self.revert)
        completer = QCompleter(self.completions)
        completer.setCaseSensitivity(Qt.CaseInsensitive)

        self.input_field.setCompleter(completer)

        # Layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.propositions_label)
        self.layout.addWidget(self.input_label)
        self.layout.addWidget(self.input_field)
        self.layout.addWidget(self.submit_button)
        self.layout.addWidget(self.revert_button)
        self.setLayout(self.layout)
        self.input_field.setFocus()
        
        if self.name:
            self.perform_lookup()
        
        else:
            self.next(action=False)
    
    def perform_lookup(self):
        self.second_run_encoded_img_path = []
        self.load_known()
        self.match_and_lookup()
        print(f"Results in {self.classified_folder}")
    
    def load_known(self):
        for self.pickle_path in self.encoded_img_paths:
            try: 
                with open(self.pickle_path, 'rb') as f:
                    self.image = pickle.load(f)
            except:
                print(f"error loading {self.pickle_path}")
                print("skipping")
                continue

            for self.face in self.image.faces:
                if self.face.name and self.face.name==self.name:
                    print(f"known photo {self.image.image_path} containing {self.face.name}")
                    self.save_face(self.face.name, action=False)
                else:
                    self.second_run_encoded_img_path.append(self.pickle_path)

    def match_and_lookup(self):
        for self.pickle_path in self.second_run_encoded_img_path:
            try: 
                with open(self.pickle_path, 'rb') as f:
                    self.image = pickle.load(f)
            except:
                print(f"error loading {self.pickle_path}")
                print("skipping")
                continue

            for self.face in self.image.faces:
                if not self.face.name:
                    self.make_propositions() # can match with one known faces
            with open(self.pickle_path, 'wb') as f:
                pickle.dump(self.image, f)

            
        
    def closeEvent(self, event):
        self.print_stats()
        return super().closeEvent(event)

    def submit(self):
        self.handle_name()
        self.input_field.setFocus()
        
    def callback_action(self):
        self.previous_pickle_path = self.pickle_path
        self.previous_face = self.face
        self.previous_index = len(self.known_faces_name)-1
    
    def handle_action(self, action):
        if action < self.k:
            self.save_face(self.propositions[action])
        elif action == self.k:
            self.save_face(UNKNOWN)
        elif action == self.k+1:
            self.unknown_for_all_faces = True
        elif action == self.k+2:
            self.save_face(BAD_QUALITY)
        elif action == self.k+3:
            self.bad_quality_for_all_faces = True
        elif action == self.k+4:
            # Not a face
            self.image.faces.remove(self.face)
            self.next()
        elif action == self.k+5:
            # Skip
            self.update_stats(skip=True)
            return self.next()
        elif action == self.k+6:
            # Skip all the faces in the image
            self.update_stats(skip_all=True)
            self.next_image = True
            return self.next()

    def handle_name(self):
        user_input = self.input_field.text()
        user_input = ' '.join(word.capitalize()
                                for word in user_input.split())
        self.save_face(user_input)

    def save_face(self, name, auto=False, action=True):
        print(f"saving {name}, action: {action}")
        self.face.name = name
        self.face.auto = auto
        self.update_stats()

        if name == BAD_QUALITY:
            return self.next(action=action)

        if self.reverting:
            self.known_faces_name[self.previous_index] = name
            self.known_faces_encoding[self.previous_index] = self.face.encoding
            self.reverting = False
        else:
            self.known_faces_name.append(name)
            self.known_faces_encoding.append(self.face.encoding)
            self.completions.add(name)

        if name==UNKNOWN:
            return self.next(action=action)

        # create dir if not exist
        folder = os.path.join(self.classified_folder, name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        # create a simlink to the image
        path = os.path.join(folder, self.image.image_path.name)
        if not os.path.islink(path):
            os.symlink(self.image.image_path, path)
        
        self.next(action=action)

    def update_stats(self, skip=False, skip_all=False, next_image=False):
        if skip_all:
            self.stats[SKIPPED] += len(self.image.faces)
            return
        if skip:
            self.stats[SKIPPED] += 1
            return
        if next_image:
            self.stats[IMAGE_COUNT] += 1
            return
        if self.face.name == BAD_QUALITY:
            self.stats[BAD_QUALITY] += 1
        elif self.face.name == UNKNOWN:
            self.stats[UNKNOWN] += 1
        if self.face.auto:
            self.stats[AUTO] += 1
    
    def print_stats(self):
        print("Stats:")
        total = 0
        for name, count in self.stats.items():
            print(f"{name}: {count}")
            total += count
        print(f"Number of faces: {total}")
        
    def next(self, action=True):
        if action:
            self.callback_action()
        self.input_field.clear()
        if self.next_image:
            self.update_stats(next_image=True)
            self.bad_quality_for_all_faces = False
            self.unknown_for_all_faces = False

            if self.pickle_path:
                # save back the image_faces with the names
                with open(self.pickle_path, 'wb') as f:
                    pickle.dump(self.image, f)

            try: 
                self.pickle_path = next(self.encoded_img_paths)
            except StopIteration:
                print("No more images !")
                self.print_stats()
                exit(0)

            try: 
                with open(self.pickle_path, 'rb') as f:
                    self.image = pickle.load(f)
            except:
                print(f"error loading {self.pickle_path}")
                print("skipping")
                self.pickle_path = None
                self.next(action=False)
                
            self.faces = iter(self.image.faces)
            self.next_image = False

        try:
            self.face = next(self.faces)
            if self.bad_quality_for_all_faces:
                self.save_face(BAD_QUALITY)
                return
            if self.unknown_for_all_faces:
                self.save_face(UNKNOWN)
        except StopIteration:
            self.next_image = True
            return self.next(action=False)

        if self.face.name:
            # if self.face.name == BAD_QUALITY or self.face.name == UNKNOWN:
            #     self.face.name = None
            # else:    
            print(f"known photo {self.image.image_path} containing {self.face.name}")
            self.save_face(self.face.name, action=False)
            return
        self.load_face()
    
    def load_face(self):
        imgdata = open(self.image.image_path, 'rb').read()
        qimage = QImage.fromData(imgdata, self.image.image_path.suffix)
        top, right, bottom, left = self.face.location
        width, heigth = right - left, bottom - top
        rect = QRect(left, top, width, heigth)
        qimage = qimage.copy(rect)
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.make_propositions()

    def revert(self):
        if not self.previous_pickle_path:
            print("No previous action")
            return
        print(f"Reverting {self.previous_pickle_path}")
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self.image, f)
        self.pickle_path = self.previous_pickle_path
        try: 
            with open(self.pickle_path, 'rb') as f:
                self.image = pickle.load(f)
        except:
            print(f"error loading {self.pickle_path}")
            print("skipping")
            self.pickle_path = None
            return self.next(action=False)
                
        self.next_image = False
        for face in self.image.faces:
            if (face.encoding == self.previous_face.encoding).all():
                self.face = face # need to point to the object belong to the Image object
                break
        self.load_face()

        self.previous_pickle_path = None
        self.previous_face = None
        self.reverting = True
        
        self.input_field.setFocus()

    def make_propositions(self):
        if not self.known_faces_encoding:
            self.render_propositions()
            return
        self.distances = face_recognition.face_distance(
            self.known_faces_encoding,
            self.face.encoding
        )
        # reverse sort and select the k lowest distances
        # TODO: optimize
        matcher = -np.argsort(-self.distances) # reverse sort
        proposer = matcher[:self.k]
        self.distances = self.distances[proposer]
        if self.distances[0] < self.threshold:
            name = self.known_faces_name[proposer[0]]
            print(f"skipped photo {self.image.image_path} containing {name} with distance {self.distances[0]}")
            self.save_face(name, auto=True, action=False)
        else:
            self.propositions = [
                self.known_faces_name[i] for i in proposer
            ]
            self.render_propositions()

    def render_propositions(self):
        propositions_lines = [f"{i} : {prop} ({score})" for i,
                        (prop, score) in enumerate(zip(self.propositions, self.distances))]
        propositions_lines.append(f"{self.k} : Unknown")
        propositions_lines.append(f"{self.k+1} : Unknown for all faces in the image")
        propositions_lines.append(f"{self.k+2} : Bad quality")
        propositions_lines.append(f"{self.k+3} : Bad quality for all faces in the image")
        propositions_lines.append(f"{self.k+4} : Not a face")
        propositions_lines.append(f"{self.k+5} : Skip")
        propositions_lines.append(f"{self.k+6} : Skip all faces in the image")
        self.propositions_label.setText(
            "\n".join(propositions_lines)
        )

    def init_completions(self, list_encoded_img_paths, contacts: Iterable|None):
        self.completions = set()
        if contacts:
            self.completions.update(contacts)
        for encoded_img_path in list_encoded_img_paths:
            with open(encoded_img_path, 'rb') as f:
                image = pickle.load(f)
                for face in image.faces:
                    if face.name and face.name!=BAD_QUALITY and face.name!=UNKNOWN:
                        self.completions.add(face.name)

class Editor(QLineEdit):
    def __init__(self):
        super().__init__()

    def clear(self):
        super().clear()        
        self.completer().setModel(QStringListModel(self.parent().completions))

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        text = event.text()
        if text.isdigit():
            self.parent().handle_action(int(text))
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.parent().handle_name()


def run_name_selector(classified_folder, encoded_img_folder, contacts):
    app = QApplication(sys.argv)
    window = NameSelector(classified_folder, encoded_img_folder, contacts)
    window.show()
    sys.exit(app.exec_())

def perform_lookup(classified_folder, encoded_img_folder, name):
    app = QApplication(sys.argv)
    window = NameSelector(classified_folder, encoded_img_folder, None, name=name)
    window.show()
    sys.exit(app.exec_())



def main():
    parser = argparse.ArgumentParser(
        description='Detect Faces')
    parser.add_argument(
        '-f', '--folder', help='Picture folders', type=str, default='demo')
    parser.add_argument(
        '-s', '--skip', help='Skip encoding and perform name attribution', action='store_true')
    parser.add_argument(
        '-r', '--restart', help='Restart the classification', action='store_true')
    parser.add_argument(
        '-c', '--contact', help='Contact file path to look for names', type=str, default=None
        )
    parser.add_argument(
        '-e', '--eval', help='Use the training to look for someone', type=str, default=None
    )
    args = parser.parse_args()
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    folder = Path(args.folder)
    classified_folder = None
    if not classified_folder:
        temp_dir = Path(tempfile.mkdtemp())
        classified_folder =    temp_dir/Path("classified")
    print(f"classified photos: {classified_folder}")
    if args.restart:
        # select the files of every subfolder and remove them using pathlib
        for file in classified_folder.glob("*/*"):
            os.remove(file)

    if not args.skip:
        face_detector = FaceDetector(folder, out_folder)
        face_detector.detect_faces()
    
    name = args.eval

    contact_file_path = args.contact
    contacts = None
    if contact_file_path:
        contacts = CSV(contact_file_path).contacts

    encoded_img_folder = Path(out_folder)
    if args.eval:
        perform_lookup(classified_folder, encoded_img_folder, name)
    else:
        run_name_selector(classified_folder, encoded_img_folder, contacts)

if __name__ == "__main__":
    main()

# python.exe .\script.py --folder="C:\Users\bcbav\Downloads\print_photos"
