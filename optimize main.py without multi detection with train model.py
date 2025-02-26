import cv2
import numpy as np
import sqlite3
import os
import pickle
from tensorflow.keras import backend as K
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import logging

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# === Model Initialization ===
def load_facenet():
    """Load the FaceNet model using keras-facenet"""
    try:
        facenet = FaceNet()
        return facenet
    except Exception as e:
        print(f"Error loading FaceNet model: {str(e)}")
        return None

def initialize_detector():
    """Initialize MTCNN detector with default settings"""
    try:
        detector = MTCNN()
        return detector
    except Exception as e:
        print(f"Error initializing MTCNN detector: {str(e)}")
        return None

# === Load Trained Model ===
def load_trained_model(model_path='face_model.pkl'):
    """
    Load the trained SVM model and label encoder from the file.
    """
    try:
        with open(model_path, 'rb') as f:
            model, label_encoder = pickle.load(f)
        return model, label_encoder
    except Exception as e:
        print(f"Error loading the trained model: {e}")
        return None, None

# === Face Embedding Generation ===
def get_face_embedding(model, face_img):
    """Generate and normalize face embedding"""
    try:
        if face_img.shape[0] < 160 or face_img.shape[1] < 160:
            face_img = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_AREA)
        
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.expand_dims(face_img, axis=0)
        embeddings = model.embeddings(face_img)
        embedding = embeddings[0] / np.linalg.norm(embeddings[0])
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

# === Camera Initialization ===
def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    return cap

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

# === Face Detection and Feature Extraction (Single Face Only) ===
def detect_and_extract_features(frame, detector, facenet_model):
    if frame is None:
        return None, None, None, None
    
    frame_with_box = frame.copy()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_image)
    
    if not faces:
        cv2.putText(frame_with_box, "No face detected", (10, 30), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, None, None, frame_with_box
    
    face = faces[0]  # Only process the first face
    if face['confidence'] < 0.9:
        cv2.putText(frame_with_box, "Low confidence", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, None, None, frame_with_box
    
    x, y, w, h = face['box']
    x_min, y_min = max(0, x), max(0, y)
    x_max, y_max = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    
    face_roi = frame[y_min:y_max, x_min:x_max]
    facenet_embedding = get_face_embedding(facenet_model, face_roi)
    
    cv2.rectangle(frame_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
    cv2.putText(frame_with_box, "Face detected", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return facenet_embedding, face_roi, (x_min, y_min, x_max, y_max), frame_with_box

# === Face Recognition ===
def recognize_face():
    # Load the trained model and label encoder
    model, label_encoder = load_trained_model('face_model.pkl')
    if model is None or label_encoder is None:
        print("Failed to load the trained model. Exiting...")
        return
    
    # Initialize MTCNN detector and FaceNet model
    detector = initialize_detector()
    facenet_model = load_facenet()
    if detector is None or facenet_model is None:
        print("Failed to initialize detector or FaceNet model. Exiting...")
        return

    # Initialize the camera
    cap = initialize_camera()

    print("Starting face recognition. Press 'q' to stop...")
    cv2.namedWindow("Recognition", cv2.WINDOW_NORMAL)

    while True:
        frame = capture_frame(cap)
        if frame is None:
            continue

        # Detect faces and extract embeddings
        facenet_embedding, _, _, frame_with_box = detect_and_extract_features(frame, detector, facenet_model)

        if facenet_embedding is not None:
            # Predict user ID using the trained model
            prediction = model.predict([facenet_embedding])[0]
            confidence = np.max(model.predict_proba([facenet_embedding]))

            # Decode the predicted label
            user_id = label_encoder.inverse_transform([prediction])[0]

            # Display the result
            label = f"Recognized: {user_id} ({confidence:.3f})" if confidence > 0.75 else "Unknown face"
            color = (0, 255, 0) if confidence > 0.75 else (0, 0, 255)
            cv2.putText(frame_with_box, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Recognition", frame_with_box)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Main Menu ===
def main_menu():
    while True:
        print("\n=== Main Menu ===")
        print("1. Register a new user")
        print("2. Start face recognition")
        print("3. View registered users")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            user_name = input("Enter the name of the user to register: ").strip()
            register_user(user_name)
        elif choice == '2':
            recognize_face()
        elif choice == '3':
            view_registered_users()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

# === View Registered Users ===
def view_registered_users():
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, registration_date FROM users')
    users = cursor.fetchall()
    conn.close()
    
    if not users:
        print("No users registered.")
        return
    
    while True:
        print("\nRegistered Users:")
        for i, (user_id, reg_date) in enumerate(users, 1):
            print(f"{i}. {user_id} (registered: {reg_date})")
        print("\nOptions:")
        print("1. Remove a user")
        print("2. Back to main menu")
        
        choice = input("Enter your choice (1-2): ").strip()
        
        if choice == '1':
            user_num = input("Enter user number to remove: ").strip()
            try:
                user_idx = int(user_num) - 1
                if 0 <= user_idx < len(users):
                    user_to_remove = users[user_idx][0]
                    confirm = input(f"Are you sure you want to remove '{user_to_remove}'? (y/n): ").lower()
                    if confirm == 'y':
                        conn = sqlite3.connect('face_database.db')
                        cursor = conn.cursor()
                        cursor.execute('DELETE FROM users WHERE id = ?', (user_to_remove,))
                        conn.commit()
                        conn.close()
                        print(f"User '{user_to_remove}' removed successfully.")
                        return
                else:
                    print("Invalid user number.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        elif choice == '2':
            return
        else:
            print("Invalid choice. Please try again.")

# === Entry Point ===
if __name__ == "__main__":
    main_menu()