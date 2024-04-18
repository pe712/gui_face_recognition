import subprocess
from PyQt5.QtCore import QStringListModel, Qt, QThread
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCompleter, QFileDialog, QCheckBox, QLayout, QPlainTextEdit
import sys
from pathlib import Path
from PyQt5.QtCore import QRect
import tempfile
from classification import FaceClassifier, FaceResetter, SpecialNames
import logging
from itertools import chain

from face import MuliprocessFaceDetector
from contacts import CSV
from PyQt5.QtTest import QTest

logger = logging.getLogger("NameSelector")

class Action:
    def __init__(self, action, label, callback):
        self.action = action
        self.label = label
        self.callback = callback
    
    def __eq__(self, __value) -> bool:
        return self.action == __value

    def __call__(self)->bool:
        return self.callback()
        
    def __str__(self):
        return f"{self.action} : {self.label}"

    def __hash__(self) -> int:
        return hash(self.action)
        

class QLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = QPlainTextEdit(parent)
        self.widget.setReadOnly(True)
        self.formatter = logging.Formatter("{asctime} > {message}", style='{', datefmt="%H:%M:%S") # {  name}

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)

class Editor(QLineEdit):
    def __init__(self, parent):
        super().__init__(parent)
        completer = QCompleter()
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.setCompleter(completer)

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

class Looker(Editor):
    def keyPressEvent(self, event):
        QLineEdit.keyPressEvent(self, event)
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.parent().perform_lookup()

class NameSelector(QWidget):
    def __init__(self):
        super().__init__()

        # TODO set height width to 600*600
        
        temp_dir = Path(tempfile.mkdtemp())
        self.classified_folder = temp_dir/Path("classified")
        self.encodings_folder = Path("encodings")
        self.contacts_folder = Path("contacts")
                
        self.k = 3
        self.face_classifier = FaceClassifier(self.classified_folder, self.encodings_folder, self.k)
        
        actions_params = [
            ("Unknown", lambda: self.face_classifier.save_face(SpecialNames.UNKNOWN)),
            ("Unknown for all faces in the image", lambda: self.face_classifier.save_face(SpecialNames.UNKNOWN, all_faces=True)),
            ("Bad quality", lambda: self.face_classifier.save_face(SpecialNames.BAD_QUALITY)),
            ("Bad quality for all faces in the image", lambda: self.face_classifier.save_face(SpecialNames.BAD_QUALITY, all_faces=True)),
            ("Not a face", self.face_classifier.remove_face),
            ("Skip", self.face_classifier.skip_face),
            ("Skip all faces in the image", lambda: self.face_classifier.skip_face(all_faces=True)),
        ]
        self._actions = [Action(self.k + i, label, callback) for i, (label, callback) in enumerate(actions_params)]
        self.initialize()
        self.set_main_layout()
    
    def initialize(self):
        folders = [self.classified_folder, self.encodings_folder, self.contacts_folder]
        for folder in folders:
            if not folder.exists():
                folder.mkdir()
        for file in self.contacts_folder.glob("*.csv"):
            contacts = CSV(file).contacts
            n = self.face_classifier.add_contacts(contacts)
            logger.debug(f"{n} contacts successfully added")
    
    def set_main_layout(self):
        add_contacts_button = QPushButton("Add contacts")
        add_contacts_button.clicked.connect(self.add_contacts)
        generate_encodings_button = QPushButton("Detect faces and generate encodings")
        generate_encodings_button.clicked.connect(self.generate_encodings)
        classify_faces_button = QPushButton("Classify faces")
        classify_faces_button.clicked.connect(self.classify_faces)
        lookup_button = QPushButton("Lookup for someone")
        lookup_button.clicked.connect(self.lookup)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)

        main_layout_widgets = [
        add_contacts_button,
        generate_encodings_button,
        classify_faces_button,
        lookup_button,
        reset_button,
        close_button,
        ]
        
        self.resetLayout()
            
        for widget in main_layout_widgets:
            self.layout.addWidget(widget)
    
    def add_research_widget(self, field_class=Editor):
        self.input_field = field_class(self)
        self.layout.addWidget(self.input_field)
        self.input_field.clear()
        self.input_field.setFocus()

    def resetLayout(self, new_layout_class=QVBoxLayout):
        # change parent of current layout to a temporary
        if isinstance(self.layout, QLayout):
            QWidget().setLayout(self.layout)
        self.layout = new_layout_class(self)

    def add_contacts(self):
        self.contact_file_button = QPushButton("Select a file with contacts")
        self.contact_file_button.setToolTip("CSV file exported from Google contacts")  
        self.contact_file_button.clicked.connect(self.get_contact_file)
        return_button = QPushButton("Return")
        return_button.clicked.connect(self.set_main_layout)
        self.resetLayout()
        self.layout.addWidget(self.contact_file_button)
        self.layout.addWidget(return_button)

    def get_contact_file(self):
        contact_file_path = Path(QFileDialog.getOpenFileName(self, "Select a file with contacts")[0])
        self.contact_file_button.setEnabled(False)
        self.contacts = CSV(contact_file_path).contacts
        save_path = self.contacts_folder/contact_file_path.name
        save_path.write_bytes(contact_file_path.read_bytes())
        
        contact_file_label = QLabel(contact_file_path.name)
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.perform_add_contacts)        
        self.layout.addWidget(contact_file_label)
        self.layout.addWidget(submit_button)
    
    def perform_add_contacts(self):
        n = self.face_classifier.add_contacts(self.contacts)
        self.set_main_layout()
        logger.debug(f"{n} contacts successfully added")

    def generate_encodings(self):
        image_folder_button = QPushButton("Select a folder with images")
        image_folder_button.clicked.connect(self.get_image_folder)
        self.recurse_image_folder = QCheckBox("Recurse subfolders ?")
        self.recurse_image_folder.setChecked(True)
        return_button = QPushButton("Return")
        return_button.clicked.connect(self.set_main_layout)
        self.resetLayout()
        self.layout.addWidget(image_folder_button)
        self.layout.addWidget(self.recurse_image_folder)
        self.layout.addWidget(return_button)
    
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
            self.image_paths = chain(self.image_folder.rglob("*.jpg"), self.image_folder.rglob("*.png"), self.image_folder.rglob("*.jpeg"))
        else:
            self.image_paths = chain(self.image_folder.glob("*.jpg"), self.image_folder.glob("*.png"), self.image_folder.glob("*.jpeg"))
        args = [(image_path, self.encodings_folder) for image_path in self.image_paths]

        self.thread = QThread(self)
        self.worker = MuliprocessFaceDetector(args)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
        info_label = QLabel("Detecting faces...\nThis may take a while\nUser intervention is not required")
        stop_button = QPushButton("Return")
        stop_button.clicked.connect(self.stop_encodings)
        
        self.resetLayout()
        self.layout.addWidget(info_label)
        self.layout.addWidget(stop_button)

    def stop_encodings(self):
        self.worker.deleteLater()
        self.thread.terminate()
        self.set_main_layout()
    
    def lookup(self):
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.perform_lookup)
        return_button = QPushButton("Return")
        return_button.clicked.connect(self.set_main_layout)

        self.resetLayout()
        self.add_research_widget(field_class=Looker)
        self.layout.addWidget(submit_button) 
        self.layout.addWidget(return_button)

    def perform_lookup(self):
        name = self.input_field.text()
        self.info_label = QLabel("Looking up for someone...\nThis may take a while\nUser intervention is not required")
        self.resetLayout()
        self.layout.addWidget(self.info_label)
        
        result_folder = self.face_classifier.lookup(name)
        # ensure the folder exists even if it is empty
        if not result_folder.exists():
            result_folder.mkdir()
        subprocess.run(["explorer", str(result_folder)])
        n = len(list(result_folder.glob("*")))
        logger.debug(f"There are {n} matchs")
        self.set_main_layout()

    def reset(self):
        check_label = QLabel("Encodings will not be deleted. This will delete every known classification. Do you want to proceed ?")
        check_button = QPushButton("Yes")
        check_button.clicked.connect(self.perform_reset)
        return_button = QPushButton("Return")
        return_button.clicked.connect(self.set_main_layout)
        self.resetLayout()
        self.layout.addWidget(check_label)
        self.layout.addWidget(check_button)
        self.layout.addWidget(return_button)
    
    def perform_reset(self):
        self.thread = QThread(self)        
        self.worker = FaceResetter(self.face_classifier)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

        info_label = QLabel("Resetting...\nThis may take a while\nUser intervention is not required.")
        return_button = QPushButton("Return")
        return_button.clicked.connect(self.set_main_layout)

        self.resetLayout()
        self.layout.addWidget(info_label)
        self.layout.addWidget(return_button)

    def closeEvent(self, event):
        stats_label = QLabel(self.get_stats())
        self.resetLayout()
        self.layout.addWidget(stats_label)
        logger.debug("Closing...")
        QTest.qWait(7000)
        super().closeEvent(event)
        self.parent().close()
    
    def get_stats(self):
        stats = self.face_classifier.get_stats()
        display = ""
        for key, value in stats.items():
            display += f"{key}: {value}\n"
        return display

    def classify_faces(self):
        self.image_label = QLabel()
        self.propositions_label = QLabel()
        input_label = QLabel("Enter name or number:")
        revert_button = QPushButton("Revert")
        revert_button.clicked.connect(self.revert_classification)
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.submit_name)
        return_button = QPushButton("Return")
        return_button.clicked.connect(self.set_main_layout)
        
        self.resetLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.propositions_label)
        self.layout.addWidget(input_label)
        self.add_research_widget()
        self.layout.addWidget(revert_button)
        self.layout.addWidget(submit_button)
        self.layout.addWidget(return_button)
        self.input_field.setFocus()
        
        self.face_classifier.load_known_names()
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
        else:
            success = self._actions[action - self.k]()
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
        propositions_lines.extend([str(action) for action in self._actions])
        self.propositions_label.setText(
            "\n".join(propositions_lines)
        )

class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('NameSelector')
        self.setGeometry(100, 100, 600, 800)

        self.q_logger = QLogger(self)
        # set root logger
        logging.getLogger().addHandler(self.q_logger)
        logging.getLogger().setLevel(logging.DEBUG)
        layout = QVBoxLayout()
        layout.addWidget(NameSelector())
        layout.addWidget(self.q_logger.widget)
        self.setLayout(layout)
        self.show()



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()