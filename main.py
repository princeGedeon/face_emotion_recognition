import cv2

# Load the Haar cascade classifier for frontal face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize video capture object from webcam (index 0)
cap = cv2.VideoCapture("Mr Bean Emotions.mp4")

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Check if frame is read correctly
  if not ret:
    break

  # Convert frame to grayscale for better detection
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect faces in the grayscale frame
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

  # Print the number of faces detected
  print(len(faces))

  # Print details of each detected face (x, y, width, height)
  print(faces)

  # Draw a rectangle around each detected face
  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

  # Display the resulting frame with detected faces
  cv2.imshow('Visages détectés' ,frame)

  # Exit loop if 'q' key is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release capture object and close all windows
cap.release()
cv2.destroyAllWindows()