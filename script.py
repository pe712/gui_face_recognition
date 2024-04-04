from PyQt5.QtCore import QStringListModel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCompleter
import sys
from pathlib import Path
import argparse
import os
from PyQt5.QtCore import QRect
import tempfile
from classification import BAD_QUALITY, UNKNOWN, FaceClassifier

from face import FaceDetector
from contacts import CSV

encodings_folder = Path("encodings")

class NameSelector(QWidget):
    def __init__(self, face_classifier:FaceClassifier):
        super().__init__()

        self.face_classifier = face_classifier
        self.k = self.face_classifier.k

        # Set up GUI components
        self.image_label = QLabel()
        self.input_label = QLabel("Enter name or number:")
        self.input_field = Editor()
        self.submit_button = QPushButton("Submit")
        self.revert_button = QPushButton("Revert")
        self.stats_label = QLabel()

        self.propositions_label = QLabel()
        self.submit_button.clicked.connect(self.submit)
        self.revert_button.clicked.connect(self.revert)
        completer = QCompleter(self.face_classifier.known_names)
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
        self.init_screen()
    
    def init_screen(self):
        # one of the two
        name = None
        if name:
            self.face_classifier.perform_lookup(name)
        else:
            self.face_classifier.next(action=False)
            self.display_face()

    def closeEvent(self, event):
        self.display_stats()
        return super().closeEvent(event)

    def submit(self):
        self.handle_name()
        self.input_field.setFocus()
    
    def revert(self):
        self.face_classifier.revert()
        self.input_field.setFocus()
        self.display_face()

    def handle_action(self, action):
        if action < self.k:
            self.face_classifier.save_face(self.face_classifier.propositions[action])
        elif action == self.k:
            self.face_classifier.save_face(UNKNOWN)
        elif action == self.k+1:
            self.face_classifier.save_face(UNKNOWN, all_faces=True)
        elif action == self.k+2:
            self.face_classifier.save_face(BAD_QUALITY)
        elif action == self.k+3:
            self.face_classifier.save_face(BAD_QUALITY, all_faces=True)
        elif action == self.k+4:
            self.face_classifier.remove_face()
        elif action == self.k+5:
            self.face_classifier.skip_face()
        elif action == self.k+6:
            self.face_classifier.skip_face(all_faces=True)
        self.input_callback()

    def handle_name(self):
        user_input = self.input_field.text()
        user_input =[word.capitalize() for word in user_input.split()]
        user_input = ' '.join(user_input)
        self.face_classifier.save_face(user_input)
        self.input_callback()

    def input_callback(self):
        self.input_field.clear()
        self.display_face()

    def display_stats(self):
        stats = self.face_classifier.get_stats()
        for key, value in stats.items():
            display = f"{key}: {value}"
        self.stats_label.setText(display)

    def display_face(self):
        imgdata = open(self.face_classifier.image.image_path, 'rb').read()
        qimage = QImage.fromData(imgdata, self.face_classifier.image.image_path.suffix)
        top, right, bottom, left = self.face_classifier.face.location
        width, heigth = right - left, bottom - top
        rect = QRect(left, top, width, heigth)
        qimage = qimage.copy(rect)
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.render_propositions()

    def render_propositions(self):
        propositions_lines = [f"{i} : {prop} ({score})" for i,
                        (prop, score) in enumerate(zip(self.face_classifier.propositions, self.face_classifier.distances))]
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


class Editor(QLineEdit):
    def __init__(self):
        super().__init__()

    def clear(self):
        super().clear()        
        self.completer().setModel(QStringListModel(self.parent().face_classifier.known_names))

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        text = event.text()
        if text.isdigit():
            self.parent().handle_action(int(text))
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.parent().handle_name()

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
    name = args.eval
    contact_file_path = args.contact

    if not os.path.isdir(encodings_folder):
        os.makedirs(encodings_folder)
    
    image_folder = Path(args.folder)
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
        face_detector = FaceDetector(image_folder, encodings_folder)
        face_detector.detect_faces()
    
    contacts = None
    if contact_file_path:
        contacts = CSV(contact_file_path).contacts

    face_classifier = FaceClassifier(classified_folder, encodings_folder, contacts)
    app = QApplication(sys.argv)
    window = NameSelector(face_classifier)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

# python.exe .\script.py --folder="C:\Users\bcbav\Downloads\print_photos"
