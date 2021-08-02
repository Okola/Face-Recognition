import face_recognition

image_of_foden = face_recognition.load_image_file('./img/known/foden.jpg')
foden_face_encoding = face_recognition.face_encodings(image_of_foden)[0]

unknown_image = face_recognition.load_image_file('./img/unknown/chiesa.jpg')
unkown_face_uncoding = face_recognition.face_encodings(unknown_image)[0]


#compare faces
result = face_recognition.compare_faces([foden_face_encoding], unkown_face_uncoding)

if result[0]:
    print('This is foden')
else:
    print('This is not Foden')