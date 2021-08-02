import face_recognition

image = face_recognition.load_image_file('./img/squad/england.jpg') #loads an image as an numpy array
face_locations = face_recognition.face_locations(image)

#Array of coodinates of each face
#print(face_locations)

print(f'There are {len(face_locations)} people') # getting the number of people in an image
