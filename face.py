import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
# Lire l'image et convertir en niveaux de gris
image = cv2.imread('data/a.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Détecter les visages
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# Dessiner des rectangles autour des visages
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Visages détectés', image)
cv2.waitKey(0)
cv2.destroyAllWindows()