import face_recognition
known_image = face_recognition.load_image_file("C:\\Users\\bcbav\\Downloads\\print_photos\\large_1564-j1_1684744503-1507ad2.jpg")
unknown_image = face_recognition.load_image_file("C:\\Users\\bcbav\\Downloads\\print_photos\\IMG-20230330-WA0007.jpg")
unknown_image2 = face_recognition.load_image_file("demo/barack2.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
unknown_encoding2 = face_recognition.face_encodings(unknown_image2)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
print(results)

results = face_recognition.face_distance([biden_encoding], unknown_encoding)
print(results)

results = face_recognition.compare_faces([biden_encoding], unknown_encoding2)
print(results)

results = face_recognition.face_distance([biden_encoding], unknown_encoding2)
print(results)