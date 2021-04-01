import numpy as np
import face_recognition as fr
import cv2

video_capture = cv2.VideoCapture(0)

sadique_image = fr.load_image_file("C:\\Users\\sadiq\\PycharmProjects\\h\\known\\sadique.jpg")
sadique_face_encoding = fr.face_encodings(sadique_image)[0]

bhuvanesh_image = fr.load_image_file("C:\\Users\\sadiq\\PycharmProjects\\h\\known\\bhuvanesh.jpg")
bhuvanesh_face_encoding = fr.face_encodings(bhuvanesh_image)[0]

sathish_image = fr.load_image_file("C:\\Users\\sadiq\\PycharmProjects\\h\\known\\sathish.jpg")
sathish_face_encoding = fr.face_encodings(sathish_image)[0]

barath_image = fr.load_image_file("C:\\Users\\sadiq\\PycharmProjects\\h\\known\\barath.jpg")
barath_face_encoding = fr.face_encodings(barath_image)[0]

known_face_encodings = [sadique_face_encoding, bhuvanesh_face_encoding, sathish_face_encoding, barath_face_encoding]
known_face_names = ["sadique", "bhuvanesh", "sathish", "barath"]

while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
