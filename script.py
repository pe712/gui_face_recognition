import time
from PyQt5.QtCore import QStringListModel, Qt, QThreadPool
from PyQt5.QtGui import QPixmap, QImage, QCloseEvent
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCompleter, QFileDialog, QCheckBox, QLayout
import sys
from pathlib import Path
from PyQt5.QtCore import QRect
import tempfile
from classification import BAD_QUALITY, UNKNOWN, FaceClassifier

from face import FaceDetector, NotThreadedFaceDetector
from contacts import CSV


class NameSelector(QWidget):
    def __init__(self):
        super().__init__()

        # TODO set height width to 600*600
        
        temp_dir = Path(tempfile.mkdtemp())
        self.classified_folder = temp_dir/Path("classified")
        self.encodings_folder = Path("encodings")

        self.k = 3
        self.face_classifier = FaceClassifier(self.classified_folder, self.encodings_folder, self.k)

        # self.logs_field = QLabel() # TODO

        self.set_main_layout()
    
    def set_main_layout(self):
        self.add_contacts_button = QPushButton("Add contacts")
        self.add_contacts_button.clicked.connect(self.add_contacts)
        self.generate_encodings_button = QPushButton("Detect faces and generate encodings")
        self.generate_encodings_button.clicked.connect(self.generate_encodings)
        self.classify_images_button = QPushButton("Classify images")
        self.classify_images_button.clicked.connect(self.classify_images)
        self.lookup_button = QPushButton("Lookup for someone")
        self.lookup_button.clicked.connect(self.lookup)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)

        main_layout_widgets = [
        self.add_contacts_button,
        self.generate_encodings_button,
        self.classify_images_button,
        self.lookup_button,
        self.close_button,
        ]
        
        self.resetLayout()
            
        for widget in main_layout_widgets:
            self.layout.addWidget(widget)
    
    def add_research_widget(self):
        self.input_field = Editor()
        completer = QCompleter(self.face_classifier.known_names)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.input_field.setCompleter(completer)
        self.layout.addWidget(self.input_field)
        self.input_field.setFocus()

    def resetLayout(self, new_layout_class=QVBoxLayout):
        # change parent of current layout to a temporary
        if isinstance(self.layout, QLayout):
            QWidget().setLayout(self.layout)
        self.layout = new_layout_class(self)

    def add_contacts(self):
        contact_file_button = QPushButton("Select a file with contacts")
        contact_file_button.clicked.connect(self.get_contact_file)

        self.resetLayout()
        self.layout.addWidget(contact_file_button)

    def get_contact_file(self):
        contact_file_path = QFileDialog.getOpenFileName(self, "Select a file with contacts")[0]
        self.contacts = CSV(contact_file_path).contacts
        contact_file_label = QLabel(contact_file_path)
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.perform_add_contacts)
        self.layout.addWidget(contact_file_label)
        self.layout.addWidget(submit_button)

    def perform_add_contacts(self):
        n = self.face_classifier.add_contacts(self.contacts)
        self.set_main_layout()
        print(f"{n} contacts successfully added")

    def generate_encodings(self):
        image_folder_button = QPushButton("Select a folder with images")
        image_folder_button.clicked.connect(self.get_image_folder)
        self.recurse_image_folder = QCheckBox("Recurse subfolders ?")
        self.recurse_image_folder.setChecked(True)
        self.resetLayout()
        self.layout.addWidget(image_folder_button)
        self.layout.addWidget(self.recurse_image_folder)
    
    def get_image_folder(self):
        self.image_folder = QFileDialog.getExistingDirectory(self, "Select a folder with images")
        image_folder_label = QLabel(self.image_folder)
        self.image_folder = Path(self.image_folder)
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.perform_generate_encodings)
        self.layout.addWidget(image_folder_label)
        self.layout.addWidget(submit_button)

    def perform_generate_encodings(self):
        if self.recurse_image_folder.isChecked():
            self.image_paths = self.image_folder.rglob("*")
        else:
            self.image_paths = self.image_folder.glob("*")
        info_label = QLabel("Detecting faces...\nThis may take a while\nUser intervention is not required")
        self.resetLayout()
        self.layout.addWidget(info_label)
        # pool = QThreadPool(self)
        # self.image_paths = list(self.image_paths)
        # n = len(self.image_paths)
        # for i in range(n%4):
        #     for image_path in self.image_paths[i:i+4]:
        #         runnable = FaceDetector(image_path, self.encodings_folder)
        #         pool.start(runnable)
        #     print("batch doing")
        #     pool.waitForDone()
        #     print("batch done")
        for image_path in self.image_paths:
            NotThreadedFaceDetector(image_path, self.image_folder).run()
        self.set_main_layout()

    def lookup(self):
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.perform_lookup)

        self.resetLayout()
        self.add_research_widget()
        self.layout.addWidget(submit_button)

    def perform_lookup(self):
        name = self.input_field.text()
        self.info_label = QLabel("Looking up for someone...\nThis may take a while\nUser intervention is not required")
        self.resetLayout()
        self.layout.addWidget(self.info_label)
        self.face_classifier.load_known(name=name)
        self.set_main_layout()

    def closeEvent(self, event):
        # TODO : does not work
        self.stats_label = QLabel()
        self.load_stats()
        self.resetLayout()
        self.layout.addWidget(self.stats_label)
        time.sleep(2)
        return super().closeEvent(event)
    
    def load_stats(self):
        stats = self.face_classifier.get_stats()
        for key, value in stats.items():
            display = f"{key}: {value}"
        self.stats_label.setText(display)


    def classify_images(self):
        self.image_label = QLabel()
        self.propositions_label = QLabel()
        input_label = QLabel("Enter name or number:")
        revert_button = QPushButton("Revert")
        submit_button = QPushButton("Submit")
        
        submit_button.clicked.connect(self.submit_name)
        revert_button.clicked.connect(self.revert_classification)
        
        self.resetLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.propositions_label)
        self.layout.addWidget(input_label)
        self.add_research_widget()
        self.layout.addWidget(revert_button)
        self.layout.addWidget(submit_button)
        self.input_field.setFocus()
        
        self.face_classifier.load_known(name_only=True)
        if self.face_classifier.next():
            self.display_face()
        else:
            self.exit_classification()

    def submit_name(self):
        self.handle_name()
        self.input_field.setFocus()
    
    def revert_classification(self):
        self.face_classifier.revert()
        self.input_field.setFocus()
        self.display_face()

    def handle_action(self, action):
        if action < self.k:
            success = self.face_classifier.save_face(self.face_classifier.propositions[action])
        elif action == self.k:
            success = self.face_classifier.save_face(UNKNOWN)
        elif action == self.k+1:
            success = self.face_classifier.save_face(UNKNOWN, all_faces=True)
        elif action == self.k+2:
            success = self.face_classifier.save_face(BAD_QUALITY)
        elif action == self.k+3:
            success = self.face_classifier.save_face(BAD_QUALITY, all_faces=True)
        elif action == self.k+4:
            success = self.face_classifier.remove_face()
        elif action == self.k+5:
            success = self.face_classifier.skip_face()
        elif action == self.k+6:
            success = self.face_classifier.skip_face(all_faces=True)
        if success:
            self.input_callback()
        else:
            self.exit_classification()
        
    def handle_name(self):
        user_input = self.input_field.text()
        user_input =[word.capitalize() for word in user_input.split()]
        user_input = ' '.join(user_input)
        if self.face_classifier.save_face(user_input):
            self.input_callback()
        else:
            self.exit_classification()

    def exit_classification(self):
        self.set_main_layout()

    def input_callback(self):
        self.input_field.clear()
        self.display_face()

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
    # parser = argparse.ArgumentParser(
    #     description='Detect Faces')
    # parser.add_argument(
    #     '-f', '--folder', help='Picture folders', type=str, default='demo')
    # parser.add_argument(
    #     '-s', '--skip', help='Skip encoding and perform name attribution', action='store_true')
    # parser.add_argument(
    #     '-r', '--restart', help='Restart the classification', action='store_true')
    # parser.add_argument(
    #     '-c', '--contact', help='Contact file path to look for names', type=str, default=None
    #     )
    # parser.add_argument(
    #     '-e', '--eval', help='Use the training to look for someone', type=str, default=None
    # )

    app = QApplication(sys.argv)
    window = NameSelector()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()