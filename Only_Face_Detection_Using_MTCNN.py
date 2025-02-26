import cv2
from mtcnn import MTCNN

# Initialize MTCNN detector
detector = MTCNN()

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Set frame size (Optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR to RGB (MTCNN expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector.detect_faces(rgb_frame)

    # Draw bounding boxes around detected faces
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Draw facial landmarks (eyes, nose, mouth)
        for key, point in face['keypoints'].items():
            cv2.circle(frame, point, 2, (0, 0, 255), -1)

    # Display the output
    cv2.imshow("MTCNN Face Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
