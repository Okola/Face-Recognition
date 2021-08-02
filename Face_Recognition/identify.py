import PIL.FontFile
import face_recognition
from PIL import Image, ImageDraw

image_of_austo = face_recognition.load_image_file('./img/known/austo.jpg')#loads the image from the images folder
austo_face_encoding = face_recognition.face_encodings(image_of_austo)[0] #finds the encodings of the loaded image

amanda_image = face_recognition.load_image_file("./img/known/amanda.jpg")
amanda_face_encoding = face_recognition.face_encodings(amanda_image)[0]

leah_image = face_recognition.load_image_file("./img/known/leah.jpg")
leah_face_encoding = face_recognition.face_encodings(leah_image)[0]

chege_image = face_recognition.load_image_file("./img/known/chege.jpg")
chege_face_encoding = face_recognition.face_encodings(chege_image)[0]

beth_image = face_recognition.load_image_file("./img/known/beth.jpg")
beth_face_encoding = face_recognition.face_encodings(beth_image)[0]




#Create an array of encodings and names
known_face_encodings = [
    austo_face_encoding, amanda_face_encoding, leah_face_encoding, chege_face_encoding, beth_face_encoding
]

known_face_names = [
    "Okola","Amanda","Leah","Chege","Beth",
]

#Load test image to find faces in

test_image = face_recognition.load_image_file('./img/squad/group.jpg')

#find faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image,face_locations)

#convert to PIL format
pil_image = Image.fromarray(test_image)

#create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

#loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings, ):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance= 0.5)


    name = "Unknown"

    #If match
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
    #Draw the box
    draw.rectangle(((left, top),(right,bottom)), outline = (255,0,0))

    #Draw label
    text_width, text_height = draw.textsize(name)


    draw.rectangle(((left,bottom - text_height - 10, (right,bottom))),fill =(255,0,0),
                   outline=(255,0,0))
    draw.text((left + 10, bottom - text_height - 7), name, fill=(255,255,255,255))

del draw

#Display image
pil_image.show()




