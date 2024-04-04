import pickle
import face_recognition
import numpy as np
import os
from face import Face, Image # needed for the pickle loading

UNKNOWN = "Unknown"
BAD_QUALITY = "Bad Quality"
AUTO = "Auto"
SKIPPED = "Skipped"
IMAGE_COUNT = "Image count"


class FaceClassifier:
    def __init__(self, classified_folder, encoded_img_folder, contacts, threshold=0.66):
        self.classified_folder = classified_folder
        self.encoded_img_folder = encoded_img_folder
        self.encoded_img_paths = encoded_img_folder.glob("*")

        self.contacts = contacts
        self.load_known_names()

        self.threshold = threshold

        self.known_faces_encoding = []
        self.known_faces_name = [] # there can be repetitions, it is the same order as the encoding

        self.pickle_path = None
        self.previous_pickle_path = None # the last action
        self.reverting = False
        self.propositions, self.distances = [], []

        self.stats = {BAD_QUALITY: 0, UNKNOWN: 0, AUTO: 0, SKIPPED: 0, IMAGE_COUNT: 0}

        self.k = 3
        
        self.next_image = True
        
        
    def load_known_names(self):
        self.known_names = set()
        if self.contacts:
            self.known_names.update(self.contacts)
        for self.pickle_path in self.encoded_img_folder.glob("*"):
            with open(self.pickle_path, 'rb') as f:
                self.image = pickle.load(f)
                for face in self.image.faces:
                    if face.name and face.name!=BAD_QUALITY and face.name!=UNKNOWN:
                        self.known_names.add(face.name)

    def make_propositions(self):
        if not self.known_faces_encoding:
            return # otherwise face_recognition raises an error
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
            print(f"Skipped photo {self.image.image_path.name} containing {name} with distance {self.distances[0]}")
            return self.save_face(name, auto=True, action=False)
        self.propositions = [
            self.known_faces_name[i] for i in proposer
        ]

    def perform_lookup(self):
        self.run2_encoded_img_paths = []
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
                    print(f"Known photo {self.image.image_path} containing {self.face.name}")
                    self.save_face(self.face.name, action=False)
                else:
                    self.run2_encoded_img_paths.append(self.pickle_path)

    def match_and_lookup(self):
        for self.pickle_path in self.run2_encoded_img_paths:
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
    
    def get_stats(self) -> dict:
        total = 0
        for name, count in self.stats.items():
            total += count
        self.stats["Total"] = total
        return self.stats

    def next(self, action=True):
        if action:
            self.callback_action()
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
            print(f"Known photo {self.image.image_path.name} containing {self.face.name}")
            self.save_face(self.face.name, action=False)
            return
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

        self.previous_pickle_path = None
        self.previous_face = None
        self.reverting = True

    def save_face(self, name, auto=False, action=True):
        print(f"Saving {name} inside {self.image.image_path.name}")
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
            self.known_names.add(name)

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

    def callback_action(self):
        self.previous_pickle_path = self.pickle_path
        self.previous_face = self.face
        self.previous_index = len(self.known_faces_name)-1

    def remove_face(self):
        self.image.faces.remove(self.face)
        self.next(action=False)

    def skip_face(self, all_faces=False):
        self.update_stats(skip=True)
        self.next()
        