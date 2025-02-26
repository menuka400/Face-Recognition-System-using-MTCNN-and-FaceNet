import cv2
import numpy as np
import sqlite3
import os
import time
from tensorflow.keras import backend as K
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import logging
from FaceRecognition import recognize_face  # Import recognize_face from FaceRecognition.py

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

# === User Registration (Capture Until 'q' Pressed) ===
def register_user(user_name):
    create_database()
    
    detector = initialize_detector()
    facenet_model = load_facenet()
    if detector is None or facenet_model is None:
        return False
    
    cap = initialize_camera()
    facenet_embeddings = []
    
    print(f"Registering user '{user_name}'. Align your face with the central box and press 's' to start capturing, 'q' to quit...")
    cv2.namedWindow("Registration", cv2.WINDOW_NORMAL)
    
    # Define central box (160x160 in a 320x240 frame)
    frame_width, frame_height = 320, 240
    box_size = 120
    central_box = (
        (frame_width - box_size) // 2,  # x_min
        (frame_height - box_size) // 2,  # y_min
        (frame_width + box_size) // 2,  # x_max
        (frame_height + box_size) // 2   # y_max
    )
    
    capturing = False
    capture_start_time = 0
    collected = 0
    
    while True:
        frame = capture_frame(cap)
        if frame is None:
            continue
        
        # Draw central box
        cv2.rectangle(frame, (central_box[0], central_box[1]), (central_box[2], central_box[3]), (255, 255, 255), 2)
        
        # Detect face
        facenet_embedding, face_roi, bbox, frame_with_box = detect_and_extract_features(frame, detector, facenet_model)
        
        if facenet_embedding is not None:
            # Check if face overlaps with central box
            face_x_min, face_y_min, face_x_max, face_y_max = bbox
            overlap = (
                face_x_min < central_box[2] and face_x_max > central_box[0] and
                face_y_min < central_box[3] and face_y_max > central_box[1]
            )
            
            if overlap:
                cv2.putText(frame_with_box, "Face aligned - Press 's' to capture", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if capturing:
                    elapsed_time = time.time() - capture_start_time
                    seconds_remaining = max(0, 5 - int(elapsed_time))
                    cv2.putText(frame_with_box, f"Capturing: {seconds_remaining}s", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame_with_box, f"Collected: {collected}", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Capture embedding every frame while within 5 seconds
                    if elapsed_time <= 5:
                        facenet_embeddings.append(facenet_embedding)
                        collected += 1
                    elif elapsed_time > 5:
                        capturing = False  # Stop capturing after 5 seconds
            
            else:
                cv2.putText(frame_with_box, "Align face with box", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame_with_box, "No face detected - Align face with box", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            capturing = False  # Reset capturing if face is lost
        
        cv2.imshow("Registration", frame_with_box)
        
        key = cv2.waitKey(100) & 0xFF
        if key == ord('s') and facenet_embedding is not None and overlap and not capturing:
            capturing = True
            capture_start_time = time.time()
        elif key == ord('q'):
            print("Registration cancelled.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(facenet_embeddings) < 5:
        print(f"Insufficient samples collected ({len(facenet_embeddings)}). Minimum 5 required. Registration failed.")
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
                    cursor.execute('SELECT facenet_embedding FROM users WHERE id = ?', (user_id,))
                    facenet_data = cursor.fetchone()[0]
                    conn.close()
                    
                    facenet_embedding = np.frombuffer(facenet_data, dtype=np.float32)
                    print(f"\nDetails for '{user_id}':")
                    print(f"FaceNet embedding shape: {facenet_embedding.shape}")
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

# === System Diagnostics ===
def run_diagnostics():
    print("\nRunning system diagnostics...")
    
    try:
        cap = initialize_camera()
        ret, frame = cap.read()
        if ret:
            print("✓ Camera initialization successful")
        else:
            print("✗ Camera not working properly")
        cap.release()
    except Exception as e:
        print(f"✗ Camera error: {str(e)}")
        return False
    
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
    
    try:
        detector = initialize_detector()
        if detector is not None:
            print("✓ MTCNN detector loaded successfully")
        else:
            print("✗ MTCNN detector failed to load")
            return False
    except Exception as e:
        print(f"✗ MTCNN detector error: {str(e)}")
        return False
    
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
    
    print("\nAll diagnostics passed successfully!")
    return True

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

# === Main Menu ===
def main_menu():
    print("\nInitializing data collection system...")
    if not run_diagnostics():
        print("\nSystem diagnostics failed. Please check the errors above.")
        return
    
    while True:
        print("\n=== Data Collection System ===")
        print("1. Start face recognition in real time")
        print("2. Register a new user")
        print("3. View registered users")
        print("4. Run system diagnostics")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
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
            print("\nShutting down system...")
            print("Thank you for using the Data Collection System")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

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