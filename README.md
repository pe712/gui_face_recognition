## About
This is a simple GUI application for face classification. It uses the well-known [face_recognition](https://github.com/ageitgey/face_recognition) library.

## Features
1. Add contacts: Import names from a csv file (google contact export) to facilitate the tagging process.
2. Detect faces and generate encodings: Detect faces in images and generate encodings for each face. This process can be long. It is speeded up by multiprocessing and can be paused and resumed at any time.
3. Classify faces: Tag faces using loaded names or additional. You can stop at any time, every progress is saved. You can go back of one step. There are special tags if you want to :
   - ignore the face (Skip)
   - remove the detection (Not a face)
   - tag as unknown (Unknown)
   - remove the detection because of bad quality (Bad quality)
At any point in this process, if the algorithm detects a match with a known face, it will automatically tag it.
4. Lookup for someone: You can search for a name and see all the faces tagged with this name, included the automatic tags.
5. Reset the classification: You can reset the classification at any time. It will remove all the tags and keep the encodings.

## Installation
```bash
git clone "https://github.com/pe712/gui_face_recognition"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python3 main.py
```

* Inspired from [FaceTag](https://github.com/roth-a/FaceTag)
