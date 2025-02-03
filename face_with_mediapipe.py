import cv2
import mediapipe as mp

from constants import EMOTIONS_CLASSES
from inference_without_tf import init_model, inference

# Initialisation de MediaPipe pour la détection de visages
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
model=init_model()
# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 112)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 112)

# Liste pour stocker les informations des visages
face_infos = []

with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Convertir l'image en RGB pour MediaPipe
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_detection.process(image)

    # Dessiner les résultats de la détection sur l'image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        # Récupérer les coordonnées et dimensions du visage
        x, y, w, h = int(detection.location_data.relative_bounding_box.xmin * image.shape[1]), \
                    int(detection.location_data.relative_bounding_box.ymin * image.shape[0]), \
                    int(detection.location_data.relative_bounding_box.width * image.shape[1]), \
                    int(detection.location_data.relative_bounding_box.height * image.shape[0])

        # Ajouter les informations du visage à la liste
        face_infos.append((x, y, w, h))
        face=image[y:y + h, x:x + w]
        i,pred=inference(face,model)
        print({
          EMOTIONS_CLASSES[str(i)]:j*100 for i,j in enumerate(list(pred[0]))
        })


        emo="EMOTIONS_CLASSES[str(i[0])]"
        cv2.putText(image, emo, (x, y), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 0), 3, cv2.LINE_AA)

        #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Afficher les informations du visage (exemple)
        #cv2.putText(image, f"Face {len(face_infos)}: ({x}, {y}, {w}, {h})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher l'image avec les visages détectés
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()

# Affichage des informations stockées (exemple)
print("Informations sur les visages détectés :")
for i, info in enumerate(face_infos):
    print(f"Visage {i+1}: {info}")