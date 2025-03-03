import cv2
import numpy as np
import sqlite3
import os
import time
from keras._tf_keras.keras import backend as K
from keras_facenet import FaceNet
import logging
import tracemalloc
import psutil
import GPUtil

import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

model_dir = 'facenet-tensorflow-tensorflow2-default-v2'

my_face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# === Model Initialization ===
def load_facenet():
    """Load the FaceNet model using keras-facenet"""
    try:
        facenet = FaceNet(key='20180402-114759')
        return facenet
    except Exception as e:
        print(f"Error loading FaceNet model: {str(e)}")
        return None

def initialize_detector():
    """Initialize detector placeholder (using MediaPipe instead of MTCNN)"""
    try:
        return 1
    except Exception as e:
        print(f"Error initializing detector: {str(e)}")
        return None

# === Face Embedding Generation ===
def get_face_embedding(model, face_img):
    """Generate and normalize face embedding"""
    try:
        if face_img.shape[0] < 160 or face_img.shape[1] < 160:
            face_img = cv2.resize(face_img, (160, 160))
        
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
def detect_and_extract_features(model, frame):
    if frame is None:
        return None, None, None, None
    tracemalloc.start()
    frame_with_box = frame.copy()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = my_face_detector.process(rgb_image)
    frame_height, frame_width, _ = rgb_image.shape
    
    if results.detections:
        for face in results.detections:
            face_rect = np.multiply(
                [
                    face.location_data.relative_bounding_box.xmin,
                    face.location_data.relative_bounding_box.ymin,
                    face.location_data.relative_bounding_box.width,
                    face.location_data.relative_bounding_box.height,
                ],
                [frame_width, frame_height, frame_width, frame_height],
            ).astype(int)
    else:
        cv2.putText(frame_with_box, "Low confidence", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, None, None, frame_with_box
    
    x, y, w, h = face_rect
    x_min, y_min = max(0, x), max(0, y)
    x_max, y_max = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

    face_roi = frame[y_min:y_max, x_min:x_max]
    facenet_embedding = get_face_embedding(model, face_roi)
    tracemalloc.stop()
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

# === User Registration (Front, Left with Prep, Right with Prep) ===
def register_user(user_name):
    create_database()
    
    facenet_model = load_facenet()
    
    cap = initialize_camera()
    facenet_embeddings = []
    
    print(f"Registering user '{user_name}'.")
    print("Please face the camera for 10 seconds (frontal), then follow the arrows with 3-second prep times.")
    cv2.namedWindow("Registration", cv2.WINDOW_NORMAL)
    
    phases = [
        ("Frontal Face", None, 10, False),           # Frontal, no arrow, 10s, no prep
        ("Prepare for Left", "left", 3, True),      # Prep for left, left arrow, 3s, prep mode
        ("Left Side", "left", 10, False),           # Left, left arrow, 10s, capture mode
        ("Prepare for Right", "right", 3, True),    # Prep for right, right arrow, 3s, prep mode
        ("Right Side", "right", 10, False),         # Right, right arrow, 10s, capture mode
    ]
    phase_idx = 0
    phase_start_time = time.time()
    collected_samples = 0
    
    while phase_idx < len(phases):
        frame = capture_frame(cap)
        if frame is None:
            continue
        
        facenet_embedding, face_roi, bbox, frame_with_box = detect_and_extract_features(facenet_model, frame)
        
        phase_name, arrow_direction, duration, is_prep = phases[phase_idx]
        elapsed_time = time.time() - phase_start_time
        
        # Display phase and timer
        remaining_time = max(0, duration - elapsed_time)
        cv2.putText(frame_with_box, f"Phase: {phase_name}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_box, f"Time left: {remaining_time:.1f}s", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if not is_prep:
            cv2.putText(frame_with_box, f"Samples: {collected_samples}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw arrows for left or right phases
        if arrow_direction == "left":
            cv2.arrowedLine(frame_with_box, (frame.shape[1] - 50, frame.shape[0] // 2), 
                           (frame.shape[1] - 100, frame.shape[0] // 2), (0, 255, 0), 2, tipLength=0.3)
        elif arrow_direction == "right":
            cv2.arrowedLine(frame_with_box, (50, frame.shape[0] // 2), 
                           (100, frame.shape[0] // 2), (0, 255, 0), 2, tipLength=0.3)
        
        # Collect samples only during capture phases (not prep)
        if facenet_embedding is not None and not is_prep:
            facenet_embeddings.append(facenet_embedding)
            collected_samples += 1
        
        cv2.imshow("Registration", frame_with_box)
        
        # Move to next phase after duration
        if elapsed_time >= duration:
            print(f"Completed {phase_name}.")
            phase_idx += 1
            phase_start_time = time.time()
            if phase_idx < len(phases):
                next_phase_name = phases[phase_idx][0]
                print(f"Now: {next_phase_name}")
                if phases[phase_idx][3]:  # If next phase is prep
                    print(f"Prepare to turn your face {phases[phase_idx][1]} in 3 seconds...")
        
        # Allow early exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Registration interrupted by user.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(facenet_embeddings) < 30:  # Minimum 30 samples (roughly 10 per capture phase)
        print(f"Insufficient samples collected ({len(facenet_embeddings)}). Minimum 30 required. Registration failed.")
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
    facenet_model = load_facenet()
    
    cap = initialize_camera()
    
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, facenet_embedding FROM users')
    stored_templates = {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in cursor.fetchall()}
    conn.close()
    
    if not stored_templates:
        print("No users registered in the database.")
        cap.release()
        return
    
    print("Starting continuous face recognition (exactly 5 frames per second). Press 'q' to stop...")
    cv2.namedWindow("Recognition", cv2.WINDOW_NORMAL)
    
    facenet_threshold = 0.70
    last_recognized_name = None
    greeting_printed = False
    no_face_count = 0
    last_state = None
    unknown_count = 0
    
    max_frames_per_second = 5
    target_frame_time = 1.0 / max_frames_per_second
    embeddings_batch = []
    frame_count = 0
    
    current_label = "No face detected"
    current_color = (0, 0, 255)
    
    while True:
        cycle_start_time = time.time()
        
        for _ in range(max_frames_per_second):
            frame_start_time = time.time()
            
            frame = capture_frame(cap)
            if frame is None:
                continue
            
            facenet_embedding, face_roi, bbox, frame_with_box = detect_and_extract_features(facenet_model, frame)
            frame_count += 1
            
            if facenet_embedding is not None:
                embeddings_batch.append(facenet_embedding)
            
            cv2.putText(frame_with_box, current_label, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)
            cv2.putText(frame_with_box, f"Unknown count: {unknown_count}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame_with_box, "FPS: 5.00", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Recognition", frame_with_box)
            
            elapsed_time = time.time() - frame_start_time
            sleep_time = target_frame_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        if embeddings_batch:
            avg_embedding = np.mean(embeddings_batch, axis=0)
            best_match, highest_confidence = None, -1
            for user_id, stored_embedding in stored_templates.items():
                similarity = np.dot(avg_embedding, stored_embedding)
                if similarity > highest_confidence:
                    highest_confidence = similarity
                    best_match = user_id
                print(f"Similarity with {user_id}: {similarity:.3f}")
            
            if highest_confidence > facenet_threshold:
                current_label = f"Recognized: {best_match} ({highest_confidence:.3f})"
                current_color = (0, 255, 0)
                
                if not greeting_printed:
                    print(f"Hello {best_match}, how's your day?😊")
                    last_recognized_name = best_match
                    greeting_printed = True
                    no_face_count = 0
                    last_state = 'recognized'
                    unknown_count = 0
                
                elif (last_recognized_name == best_match and 
                      no_face_count >= 5 and 
                      last_state == 'no_face'):
                    print(f"{best_match}, welcome back I miss you😍")
                    no_face_count = 0
                    last_state = 'recognized'
                    unknown_count = 0
                
            else:
                current_label = f"Unknown face ({highest_confidence:.3f})"
                current_color = (0, 0, 255)
                unknown_count += 1
                print(f"Unknown face detected. Count: {unknown_count}")
                
                if unknown_count >= 10:
                    print("Unknown face detected 10 times. Initiating registration...")
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    user_name = input("Unrecognized face detected. Please enter your name to register: ").strip()
                    if user_name:
                        success = register_user(user_name)
                        if success:
                            conn = sqlite3.connect('face_database.db')
                            cursor = conn.cursor()
                            cursor.execute('SELECT id, facenet_embedding FROM users')
                            stored_templates = {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in cursor.fetchall()}
                            conn.close()
                            print("Registration complete. Resuming recognition...")
                        else:
                            print("Registration failed. Resuming recognition...")
                    else:
                        print("No name provided. Resuming recognition...")
                    
                    cap = initialize_camera()
                    cv2.namedWindow("Recognition", cv2.WINDOW_NORMAL)
                    unknown_count = 0
        else:
            current_label = "No face detected"
            current_color = (0, 0, 255)
            if last_recognized_name is not None and greeting_printed:
                no_face_count += 1
                if no_face_count >= 5 and last_state == 'recognized':
                    print(f"{last_recognized_name}, where you go come back I need you🥺")
                    last_state = 'no_face'
                elif last_state == 'no_face':
                    no_face_count += 1
            unknown_count = 0
        
        embeddings_batch = []
        frame_count = 0
        
        cycle_elapsed_time = time.time() - cycle_start_time
        if cycle_elapsed_time < 1.0:
            time.sleep(1.0 - cycle_elapsed_time)
        print(f"Processed 5 frames in {time.time() - cycle_start_time:.2f} seconds")
    
    cap.release()
    cv2.destroyAllWindows()

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
            print("✓ Detector loaded successfully")
        else:
            print("✗ Detector failed to load")
            return False
    except Exception as e:
        print(f"✗ Detector error: {str(e)}")
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

# === Main Menu ===
def main_menu():
    print("\nInitializing system...")
    if not run_diagnostics():
        print("\nSystem diagnostics failed. Please check the errors above.")
        return
    
    while True:
        print("\n=== Lightweight Face Recognition System ===")
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
            print("Thank you for using the Face Recognition System")
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