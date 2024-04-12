import pickle
import face_recognition
import numpy as np
import os
from face import Face, Image # needed for the pickle loading

import logging
logger = logging.getLogger(__name__)

UNKNOWN = "Unknown"
BAD_QUALITY = "Bad Quality"
AUTO = "Auto"
SKIPPED = "Skipped"
IMAGE_COUNT = "Image count"
REMOVED = "Face detection removed"


class FaceClassifier:
    def __init__(self, classified_folder, encoded_img_folder, k, threshold=0.66):
        self.classified_folder = classified_folder
        self.encoded_img_folder = encoded_img_folder
        self.k = k

        self.known_names = set()

        self.threshold = threshold

        self.known_faces_encoding = []
        self.known_faces_name = [] # there can be repetitions, it is the same order as the encoding

        self.pickle_path = None
        self.propositions, self.distances = [], []
        self.face = None
        
        self.removed = False
        self.action = None # used to track the last user action to be able to revert
        self.previous_pickle_path = None
        self.reverting = False
        
        self.stats = {BAD_QUALITY: 0, UNKNOWN: 0, AUTO: 0, SKIPPED: 0, IMAGE_COUNT: 0, REMOVED: 0}
        
        self.next_image = True
    
    def add_contacts(self, contacts):
        self.known_names.update(contacts)
        return len(contacts)

    def make_propositions(self):
        if not self.known_faces_encoding:
            return # otherwise face_recognition raises an error
        self.distances = face_recognition.face_distance(
            self.known_faces_encoding,
            self.face.encoding
        )
        # reverse sort and select the k lowest distances
        # TODO: optimize
        matcher = -np.argsort(-self.distances)
        proposer = matcher[:self.k]
        self.distances = self.distances[proposer]
        if self.distances[0] < self.threshold:
            name = self.known_faces_name[proposer[0]]
            logger.debug(f"Skipped photo {self.image.image_path.name} containing {name} with distance {self.distances[0]}")
            return self.save_face(name, auto=True, action=False)
        self.propositions = [
            self.known_faces_name[i] for i in proposer
        ]
    
    def load_known_names(self):
        for self.pickle_path in self.encoded_img_folder.glob("*.pickle"):
            with open(self.pickle_path, 'rb') as f:
                self.image = pickle.load(f)

            for self.face in self.image.faces:
                if self.face.name and self.face.name and self.face.name!=BAD_QUALITY and self.face.name!=UNKNOWN:
                    logger.debug(f"Known photo {self.image.image_path} containing {self.face.name}")
                    self.known_names.add(self.face.name)
        self.encoded_img_paths = iter(self.encoded_img_folder.glob("*.pickle"))

    def lookup(self, name):
        self.encoded_img_paths = self.encoded_img_folder.glob("*.pickle")
        # first run
        self._auto_match(name)
        # second run
        self._auto_match(name)
        return self.classified_folder/name
    
    def _auto_match(self, name):
        second_run =[]
        for self.pickle_path in self.encoded_img_paths:
            with open(self.pickle_path, 'rb') as f:
                self.image = pickle.load(f)

            for self.face in self.image.faces:
                if self.face.name:
                    if self.face.name==name:
                        self.save_face(self.face.name, action=False)
                    else:
                        second_run.append(self.pickle_path)
                else:
                    self.make_propositions() # auto match and save if possible, otherwise pass 

            with open(self.pickle_path, 'wb') as f:
                pickle.dump(self.image, f)
        self.encoded_img_paths = iter(second_run)

    def update_stats(self):
        if self.removed:
            self.stats[REMOVED]+= 1
            self.removed = False
            return
        if self.face.auto:
            self.stats[AUTO] += 1
        if not self.face.name:
            if self.next_image:
                self.stats[SKIPPED] += len(self.image.faces)
            else:
                self.stats[SKIPPED] += 1
        elif self.face.name == BAD_QUALITY:
            self.stats[BAD_QUALITY] += 1
        elif self.face.name == UNKNOWN:
            self.stats[UNKNOWN] += 1
    
    def update_image_count(self):
        self.stats[IMAGE_COUNT] += 1

    def get_stats(self) -> dict:
        total = 0
        for name, count in self.stats.items():
            total += count
        self.stats["Total"] = total
        return self.stats

    def next(self)->bool:
        assert hasattr(self, 'encoded_img_paths'), "load_known must be called first"
        if self.next_image:
            self.update_image_count()
            self.bad_quality_for_all_faces = False
            self.unknown_for_all_faces = False

            if self.pickle_path:
                # save back the image_faces with the names
                with open(self.pickle_path, 'wb') as f:
                    pickle.dump(self.image, f)

            try: 
                self.pickle_path = next(self.encoded_img_paths)
            except StopIteration:
                logger.debug("No more images !")
                return False
            
            with open(self.pickle_path, 'rb') as f:
                self.image = pickle.load(f)
                
            self.faces = iter(self.image.faces)
            self.next_image = False

        try:
            self.face = next(self.faces)
            if self.bad_quality_for_all_faces:
                return self.save_face(BAD_QUALITY, all_faces=True)
            if self.unknown_for_all_faces:
                return self.save_face(UNKNOWN, all_faces=True)
        except StopIteration:
            self.next_image = True
            return self.next()

        if self.face.name:
            # if self.face.name == BAD_QUALITY or self.face.name == UNKNOWN:
            #     self.face.name = None
            # else:    
            logger.debug(f"Known photo {self.image.image_path.name} containing {self.face.name}")
            return self.save_face(self.face.name, action=False)
        self.make_propositions()
        return True
    
    def revert(self):
        if not self.action:
            logger.debug("No previous action")
            return
    
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self.image, f)
        
        self.pickle_path = self.action.pickle_path
        logger.debug(f"Reverting {self.pickle_path.name}")
        with open(self.pickle_path, 'rb') as f:
            self.image = pickle.load(f)

        self.next_image = False
        for face in self.image.faces:
            if (face.encoding == self.action.previous_face.encoding).all():
                self.face = face # need to point to the object belong to the Image object
                break

        self.previous_pickle_path = None
        self.previous_face = None
        self.reverting = True
   
    def save_face(self, name, auto=False, action=True, all_faces=False)->bool:
        logger.debug(f"Saving {name} inside {self.image.image_path.name}")
        self.face.name = name
        self.face.auto = auto
        
        if action:
            self._update_action()

        if self.reverting:
            self.known_faces_name[self.action.previous_index] = name
            self.known_faces_encoding[self.action.previous_index] = self.face.encoding
            self.reverting = False
        else:
            self.known_faces_name.append(name)
            self.known_faces_encoding.append(self.face.encoding)
            self.known_names.add(name)

        if name == BAD_QUALITY:
            self.bad_quality_for_all_faces = all_faces
            return self.next()
    
        if name==UNKNOWN:
            self.unknown_for_all_faces = all_faces
            return self.next()
        
        # create dir if not exist
        folder = os.path.join(self.classified_folder, name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        # create a simlink to the image
        path = os.path.join(folder, self.image.image_path.name)
        if not os.path.islink(path):
            os.symlink(self.image.image_path, path)
        self.update_stats()
        return self.next()

    def remove_face(self)->bool:
        self.image.faces.remove(self.face)
        self.removed = True
        self.update_stats()
        return self.next()

    def skip_face(self, all_faces=False)->bool:
        self._update_action()
        self.next_image = all_faces
        self.update_stats()
        return self.next()
    
    def _get_current_index(self):
        return len(self.known_faces_name)-1

    def _update_action(self):
        self.action = Action(self.pickle_path, self.face, self._get_current_index())

class Action:
    def __init__(self, pickle_path, face, index):
        self.pickle_path = pickle_path
        self.previous_face = face
        self.previous_index = index
