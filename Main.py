import pyrealsense2 as rs
import cv2
import numpy as np
import mediapipe as mp
import sqlite3
import time
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
from mtcnn import MTCNN
from keras_facenet import FaceNet
import logging

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')  # Only show errors
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Disable TensorFlow optimization messages
tf.config.set_visible_devices([], 'GPU')  # Disable GPU devices if any
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

def load_facenet():
    """Load the FaceNet model using keras-facenet"""
    try:
        facenet = FaceNet()
        return facenet
    except Exception as e:
        print(f"Error loading FaceNet model: {str(e)}")
        return None

def get_face_embedding(model, face_img):
    """Generate face embedding using keras-facenet"""
    try:
        if face_img.shape[0] < 160 or face_img.shape[1] < 160:
            face_img = cv2.resize(face_img, (160, 160))
            
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.expand_dims(face_img, axis=0)
        embeddings = model.embeddings(face_img)
        return embeddings[0]
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

# === Camera Initialization ===
def initialize_camera():
    cap = cv2.VideoCapture(0)  # Use default webcam (usually laptop camera)
    if not cap.isOpened():
        raise Exception("Could not open webcam")
    return cap

# === Capture Frame ===
def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

# === Face Detection and Feature Extraction ===
def detect_and_extract_features(frame, face_detection, facenet_model):
    if frame is None:
        return None, None, None, None
    
    frame_with_box = frame.copy()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)
    
    if not results.detections:
        cv2.putText(frame_with_box, "No face detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return None, None, None, frame_with_box
    
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    
    h, w, _ = frame.shape
    x_min = max(0, int(bbox.xmin * w))
    x_max = min(w, int((bbox.xmin + bbox.width) * w))
    y_min = max(0, int(bbox.ymin * h))
    y_max = min(h, int((bbox.ymin + bbox.height) * h))
    
    if x_max <= x_min or y_max <= y_min:
        return None, None, None, frame_with_box
    
    face_roi = frame[y_min:y_max, x_min:x_max]
    facenet_embedding = get_face_embedding(facenet_model, face_roi)
    
    cv2.rectangle(frame_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(frame_with_box, "Face detected", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return facenet_embedding, face_roi, (x_min, y_min, x_max, y_max), frame_with_box

# === Database Functions ===
def create_database():
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            facenet_embedding BLOB,
            registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# === User Registration ===
def register_user(user_name, num_samples=20):
    create_database()
    
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    facenet_model = load_facenet()
    
    cap = initialize_camera()
    facenet_embeddings = []
    
    print(f"Registering user '{user_name}'. Please face the camera...")
    cv2.namedWindow("Registration", cv2.WINDOW_NORMAL)
    
    collected = 0
    while collected < num_samples:
        frame = capture_frame(cap)
        if frame is None:
            continue
        
        facenet_embedding, face_roi, bbox, frame_with_box = detect_and_extract_features(
            frame, face_detection, facenet_model
        )
        
        cv2.imshow("Registration", frame_with_box)
        
        if facenet_embedding is not None:
            facenet_embeddings.append(facenet_embedding)
            collected += 1
            print(f"Captured {collected}/{num_samples} samples")
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            print("Registration interrupted")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()
    
    if len(facenet_embeddings) < 5:
        print("Insufficient samples collected. Registration failed.")
        return False
    
    mean_facenet_embedding = np.mean(facenet_embeddings, axis=0)
    
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT OR REPLACE INTO users (id, facenet_embedding) VALUES (?, ?)',
        (user_name, mean_facenet_embedding.tobytes())
    )
    conn.commit()
    conn.close()
    
    print(f"User '{user_name}' registered successfully with {len(facenet_embeddings)} samples.")
    return True

# === Face Recognition ===
def recognize_face():
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    facenet_model = load_facenet()
    
    cap = initialize_camera()
    
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, facenet_embedding FROM users')
    stored_templates = {}
    for row in cursor.fetchall():
        user_id, facenet_data = row
        stored_templates[user_id] = np.frombuffer(facenet_data, dtype=np.float32)
    conn.close()
    
    if not stored_templates:
        print("No users registered in the database.")
        cap.release()
        face_detection.close()
        return
    
    print("Starting face recognition. Press 'q' to stop...")
    cv2.namedWindow("Recognition", cv2.WINDOW_NORMAL)
    
    facenet_threshold = 0.7  # Cosine similarity threshold
    
    while True:
        frame = capture_frame(cap)
        if frame is None:
            continue
        
        facenet_embedding, face_roi, bbox, frame_with_box = detect_and_extract_features(
            frame, face_detection, facenet_model
        )
        
        if facenet_embedding is not None:
            best_match = None
            highest_confidence = -1
            
            for user_id, stored_embedding in stored_templates.items():
                similarity = np.dot(facenet_embedding, stored_embedding)
                
                if similarity > highest_confidence:
                    highest_confidence = similarity
                    best_match = user_id
            
            if highest_confidence > facenet_threshold:
                cv2.putText(frame_with_box, 
                          f"Recognized: {best_match} ({highest_confidence:.3f})",
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame_with_box,
                          f"Unknown face ({highest_confidence:.3f})",
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Recognition", frame_with_box)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()

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
        print("1. View user details")
        print("2. Remove a user")
        print("3. Back to main menu")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            user_num = input("Enter user number to view details: ").strip()
            try:
                user_idx = int(user_num) - 1
                if 0 <= user_idx < len(users):
                    user_id = users[user_idx][0]
                    conn = sqlite3.connect('face_database.db')
                    cursor = conn.cursor()
                    cursor.execute('SELECT facenet_embedding, depth_features FROM users WHERE id = ?', (user_id,))
                    facenet_data, depth_data = cursor.fetchone()
                    conn.close()
                    
                    facenet_embedding = np.frombuffer(facenet_data, dtype=np.float32)
                    depth_features = np.frombuffer(depth_data, dtype=np.float64)
                    
                    print(f"\nDetails for '{user_id}':")
                    print(f"FaceNet embedding shape: {facenet_embedding.shape}")
                    print(f"Depth features shape: {depth_features.shape}")
                else:
                    print("Invalid user number.")
            except ValueError:
                print("Please enter a valid number.")
        
        elif choice == '2':
            user_num = input("Enter user number to remove: ").strip()
            try:
                user_idx = int(user_num) - 1
                if 0 <= user_idx < len(users):
                    user_id = users[user_idx][0]
                    conn = sqlite3.connect('face_database.db')
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
                    
                    conn.commit()
                    conn.close()
                    print(f"User '{user_id}' removed.")
                    users.pop(user_idx)
                else:
                    print("Invalid user number.")
            except ValueError:
                print("Please enter a valid number.")
        
        elif choice == '3':
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# === Error Handling Wrapper ===
def safe_operation(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            print("Please try again or contact support if the issue persists.")
            return None
    return wrapper

# === System Configuration ===
def configure_system():
    config = {
        'facenet_threshold': 0.7,
        'depth_threshold': 0.5,
        'min_samples': 5,
        'recommended_samples': 20,
        'facenet_weight': 0.7,
        'depth_weight': 0.3
    }
    
    if os.path.exists('config.txt'):
        try:
            with open('config.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    key, value = line.strip().split('=')
                    config[key] = float(value)
        except:
            print("Error loading config. Using defaults.")
    
    return config

# === System Diagnostics ===
def run_diagnostics():
    print("\nRunning system diagnostics...")
    
    # Check camera
    try:
        cap = initialize_camera()
        ret, frame = cap.read()  # Try to read a frame
        if ret:
            print("✓ Camera initialization successful")
        else:
            print("✗ Camera not working properly")
        cap.release()  # Properly release the camera
    except Exception as e:
        print(f"✗ Camera error: {str(e)}")
        return False
    
    # Check FaceNet model
    try:
        model = load_facenet()
        if model is not None:
            print("✓ FaceNet model loaded successfully")
        else:
            print("✗ FaceNet model failed to load")
            return False
    except Exception as e:
        print(f"✗ FaceNet model error: {str(e)}")
        return False
    
    # Check database
    try:
        create_database()
        conn = sqlite3.connect('face_database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users')
        user_count = cursor.fetchone()[0]
        conn.close()
        print(f"✓ Database check successful ({user_count} users registered)")
    except Exception as e:
        print(f"✗ Database error: {str(e)}")
        return False
    
    # Check OpenCV
    try:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (99, 99), (255, 255, 255), 1)
        print("✓ OpenCV check successful")
    except Exception as e:
        print(f"✗ OpenCV error: {str(e)}")
        return False
    
    print("\nAll diagnostics passed successfully!")
    return True

# === Enhanced Main Menu ===
def main_menu():
    print("\nInitializing system...")
    config = configure_system()
    
    if not run_diagnostics():
        print("\nSystem diagnostics failed. Please check the errors above.")
        return
    
    while True:
        print("\n=== 3D Face Recognition System with FaceNet ===")
        print("1. Start face recognition in real time")
        print("2. Register a new user")
        print("3. View registered users")
        print("4. Run system diagnostics")
        print("5. Configure system")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            safe_operation(recognize_face)()
        
        elif choice == '2':
            user_name = input("Enter user name: ").strip()
            if user_name:
                safe_operation(register_user)(user_name)
            else:
                print("User name cannot be empty.")
        
        elif choice == '3':
            safe_operation(view_registered_users)()
        
        elif choice == '4':
            safe_operation(run_diagnostics)()
        
        elif choice == '5':
            print("\nCurrent configuration:")
            for key, value in config.items():
                print(f"{key} = {value}")
            
            print("\nTo modify settings, edit 'config.txt'")
            input("Press Enter to continue...")
        
        elif choice == '6':
            print("\nShutting down system...")
            print("Thank you for using 3D Face Recognition System")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

# === Program Entry Point ===
if __name__ == "__main__":
    try:
        if os.path.exists('face_database.db') and input("Clear existing database? (y/n): ").lower() == 'y':
            os.remove('face_database.db')
            print("Old database cleared. Please re-register users.")
        
        main_menu()
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Shutting down...")
    except Exception as e:
        print(f"\nUnexpected error occurred: {str(e)}")
        print("Please restart the program or contact support.")
    finally:
        cv2.destroyAllWindows()
