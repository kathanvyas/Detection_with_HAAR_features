# Homework Solution

# Importing the libraries
import cv2

# Loading the cascades
face_cover = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cover = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cover = cv2.CascadeClassifier('haarcascade_smile.xml')

# Defining a function that will do the detections
def detect(g, f):
    face_detect = face_cover.detectMultiScale(g, 1.3, 5)
    for (x, y, w, h) in face_detect:
        cv2.rectangle(f, (x, y), (x+w, y+h), (255, 0, 0), 2)
        for_g = g[y:y+h, x:x+w]
        for_c = f[y:y+h, x:x+w]
        eye_detect = eye_cover.detectMultiScale(for_g, 1.1, 22)
        for (ex, ey, ew, eh) in eye_detect:
            cv2.rectangle(for_c, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        smile_detect = smile_cover.detectMultiScale(for_g, 1.7, 22)
        for (sx, sy, sw, sh) in smile_detect:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    return f

# Doing some Face Recognition with the webcam
capture = cv2.VideoCapture(0)
while True:
    _, f = capture.read()
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    canvas = detect(g, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
