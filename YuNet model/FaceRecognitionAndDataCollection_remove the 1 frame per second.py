import cv2
import numpy as np
import sqlite3
import os
import time
from tensorflow.keras import backend as K
from keras_facenet import FaceNet
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
    """Initialize YuNet detector for long-distance detection"""
    try:
        detector = cv2.FaceDetectorYN.create(
            model=r"C:\Users\menuk\Desktop\Face-Recognition-System-using-BlazeFace-and-FaceNet\YuNet model\face_detection_yunet_2023mar.onnx",
            config="",  # YuNet doesn’t require a config file
            input_size=(640, 480),  # Higher resolution for long-distance detection
            score_threshold=0.7,    # Lowered for smaller/distant faces
            nms_threshold=0.3,
            top_k=5000
        )
        return detector
    except Exception as e:
        print(f"Error initializing YuNet detector: {str(e)}")
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Increased resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

# === Face Detection and Feature Extraction with YuNet ===
def detect_and_extract_features(frame, detector, facenet_model):
    if frame is None:
        return None, None, None, None
    
    frame_with_box = frame.copy()
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    
    # YuNet detection
    _, faces = detector.detect(frame)
    
    if faces is None or len(faces) == 0:
        cv2.putText(frame_with_box, "No face detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, None, None, frame_with_box
    
    # Process the first face only
    face = faces[0]
    x, y, w, h = face[:4].astype(int)  # YuNet returns [x, y, w, h, confidence, landmarks...]
    confidence = face[4]
    
    if confidence < 0.7:  # Adjusted threshold
        cv2.putText(frame_with_box, "Low confidence", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, None, None, frame_with_box
    
    x_min, y_min = max(0, x), max(0, y)
    x_max, y_max = min(width, x + w), min(height, y + h)
    
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

# === Draw Arrows ===
def draw_left_arrow(frame, position):
    """Draw a small left arrow on the frame"""
    x, y = position
    cv2.arrowedLine(frame, (x + 20, y), (x, y), (0, 255, 0), 2, tipLength=0.3)
    return frame

def draw_right_arrow(frame, position):
    """Draw a small right arrow on the frame"""
    x, y = position
    cv2.arrowedLine(frame, (x - 20, y), (x, y), (0, 255, 0), 2, tipLength=0.3)
    return frame

# === Check Face Alignment with Box ===
def is_face_aligned(face_bbox, box):
    """Check if the detected face aligns with the given box"""
    if face_bbox is None:
        return False
    
    face_x_min, face_y_min, face_x_max, face_y_max = face_bbox
    box_x_min, box_y_min, box_x_max, box_y_max = box
    
    # Check overlap with tolerance
    overlap = (
        face_x_min < box_x_max and face_x_max > box_x_min and
        face_y_min < box_y_max and face_y_max > box_y_min
    )
    
    # Additional check for centering
    face_center_x = (face_x_min + face_x_max) / 2
    face_center_y = (face_y_min + face_y_max) / 2
    box_center_x = (box_x_min + box_x_max) / 2
    box_center_y = (box_y_min + box_y_max) / 2
    
    tolerance_w = (box_x_max - box_x_min) * 0.25  # 25% of box width
    tolerance_h = (box_y_max - box_y_min) * 0.25  # 25% of box height
    
    return overlap and (abs(face_center_x - box_center_x) < tolerance_w and 
                        abs(face_center_y - box_center_y) < tolerance_h)

# === Face Recognition ===
def recognize_face():
    detector = initialize_detector()
    facenet_model = load_facenet()
    if detector is None or facenet_model is None:
        print("Failed to initialize models")
        return
    
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
    
    print("Starting face recognition. Press 'q' to stop...")
    cv2.namedWindow("Recognition", cv2.WINDOW_NORMAL)
    
    facenet_threshold = 0.75
    last_recognized_name = None
    greeting_printed = False
    no_face_count = 0
    last_state = None

    while True:
        frame_start_time = time.time()
        
        frame = capture_frame(cap)
        if frame is None:
            continue
        
        facenet_embedding, _, bbox, frame_with_box = detect_and_extract_features(frame, detector, facenet_model)
        
        # Calculate processing time and FPS (for display only, no delay enforced)
        elapsed_time = time.time() - frame_start_time
        elapsed_ms = int(elapsed_time * 1000)
        elapsed_s = int(elapsed_time) if elapsed_time >= 1 else 0
        progress_output = f"1/1 ━━━━━━━━━━━━━━━━━━━━ {elapsed_s}s {elapsed_s}s/step" if elapsed_s > 0 else f"1/1 ━━━━━━━━━━━━━━━━━━━━ 0s {elapsed_ms}ms/step"
        print(progress_output)
        
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame_with_box, fps_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if facenet_embedding is not None:
            best_match, highest_confidence = None, -1
            for user_id, stored_embedding in stored_templates.items():
                similarity = np.dot(facenet_embedding, stored_embedding)
                if similarity > highest_confidence:
                    highest_confidence = similarity
                    best_match = user_id
            
            label = (f"Recognized: {best_match} ({highest_confidence:.3f})" 
                     if highest_confidence > facenet_threshold else 
                     f"Unknown face ({highest_confidence:.3f})")
            color = (0, 255, 0) if highest_confidence > facenet_threshold else (0, 0, 255)
            cv2.putText(frame_with_box, label, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if highest_confidence > facenet_threshold and not greeting_printed:
                print(f"Hello {best_match}, how's your day?😊")
                last_recognized_name = best_match
                greeting_printed = True
                no_face_count = 0
                last_state = 'recognized'
            
            elif (highest_confidence > facenet_threshold and 
                  last_recognized_name == best_match and 
                  no_face_count >= 5 and 
                  last_state == 'no_face'):
                print(f"{best_match}, welcome back I miss you😍")
                no_face_count = 0
                last_state = 'recognized'
        
        else:
            if last_recognized_name is not None and greeting_printed:
                no_face_count += 1
                if no_face_count >= 5 and last_state == 'recognized':
                    print(f"{last_recognized_name}, where you go come back I need you🥺")
                    last_state = 'no_face'
            elif last_recognized_name is not None and greeting_printed and last_state == 'no_face':
                no_face_count += 1

        cv2.imshow("Recognition", frame_with_box)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# === User Registration (Capture Until 'q' Pressed) ===
def register_user(user_name):
    create_database()
    
    detector = initialize_detector()
    facenet_model = load_facenet()
    if detector is None or facenet_model is None:
        return False
    
    cap = initialize_camera()
    facenet_embeddings = []
    
    print(f"Registering user '{user_name}'. Align your front face with the box and press 's' to start capturing, 'q' to quit...")
    cv2.namedWindow("Registration", cv2.WINDOW_NORMAL)
    
    frame_width, frame_height = 640, 480  # Increased resolution
    
    # Front face box (320x320 centered)
    front_box_size = 320  # Larger box for long distance
    front_box = (
        (frame_width - front_box_size) // 2,  # x_min (160)
        (frame_height - front_box_size) // 2,  # y_min (80)
        (frame_width + front_box_size) // 2,  # x_max (480)
        (frame_height + front_box_size) // 2   # y_max (400)
    )
    
    # Left face box (240x320, shifted left)
    side_box_width = 240
    side_box_height = 320
    left_box = (
        (frame_width - side_box_width) // 4,  # x_min (shifted left to 100)
        (frame_height - side_box_height) // 2,  # y_min (80)
        (frame_width - side_box_width) // 4 + side_box_width,  # x_max (340)
        (frame_height + side_box_height) // 2   # y_max (400)
    )
    
    # Right face box (240x320, shifted right)
    right_box = (
        frame_width - (frame_width - side_box_width) // 4 - side_box_width,  # x_min (shifted right to 300)
        (frame_height - side_box_height) // 2,  # y_min (80)
        frame_width - (frame_width - side_box_width) // 4,  # x_max (540)
        (frame_height + side_box_height) // 2   # y_max (400)
    )
    
    # Phase 1: Front face capture
    capturing_front = False
    capture_start_time_front = 0
    collected_front = 0
    phase = "front"
    
    # Phase 2: Left face capture
    capturing_left = False
    capture_start_time_left = 0
    collected_left = 0
    
    # Phase 3: Right face capture
    capturing_right = False
    capture_start_time_right = 0
    collected_right = 0
    
    while True:
        frame = capture_frame(cap)
        if frame is None:
            continue
        
        # Draw box based on phase
        if phase == "front":
            current_box = front_box
        elif phase == "left":
            current_box = left_box
            frame_with_box = draw_left_arrow(frame_with_box, (frame_width - 80, 40))  # Adjusted for larger frame
        else:  # phase == "right"
            current_box = right_box
            frame_with_box = draw_right_arrow(frame_with_box, (40, 40))  # Adjusted for larger frame
        
        cv2.rectangle(frame, (current_box[0], current_box[1]), (current_box[2], current_box[3]), (255, 255, 255), 2)
        
        # Detect face with YuNet for embedding
        facenet_embedding, face_roi, bbox, frame_with_box = detect_and_extract_features(frame, detector, facenet_model)
        
        # Debug output to check detection
        if facenet_embedding is None:
            print("Debug: No embedding generated - YuNet failed to detect face.")
        
        aligned = is_face_aligned(bbox, current_box)
        
        if phase == "front":
            if aligned and facenet_embedding is not None:
                cv2.putText(frame_with_box, "Front face aligned - Press 's' to capture", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if capturing_front:
                    elapsed_time = time.time() - capture_start_time_front
                    seconds_remaining = max(0, 5 - int(elapsed_time))
                    cv2.putText(frame_with_box, f"Capturing front: {seconds_remaining}s", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame_with_box, f"Collected: {collected_front}", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    if elapsed_time <= 5:
                        facenet_embeddings.append(facenet_embedding)
                        collected_front += 1
                    elif elapsed_time > 5:
                        capturing_front = False
                        phase = "left"
                        print("Front face capture complete. Now turn your face to the left and press 's' to start capturing.")
            else:
                cv2.putText(frame_with_box, "Align front face with box", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                capturing_front = False
        
        elif phase == "left":
            if aligned and facenet_embedding is not None:
                cv2.putText(frame_with_box, "Left face aligned - Press 's' to capture", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if capturing_left:
                    elapsed_time = time.time() - capture_start_time_left
                    seconds_remaining = max(0, 5 - int(elapsed_time))
                    cv2.putText(frame_with_box, f"Capturing left: {seconds_remaining}s", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame_with_box, f"Collected: {collected_front + collected_left}", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    if elapsed_time <= 5:
                        facenet_embeddings.append(facenet_embedding)
                        collected_left += 1
                    elif elapsed_time > 5:
                        capturing_left = False
                        phase = "right"
                        print("Left face capture complete. Now turn your face to the right and press 's' to start capturing.")
            else:
                cv2.putText(frame_with_box, "Turn face left to align with box", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                capturing_left = False
        
        elif phase == "right":
            if aligned and facenet_embedding is not None:
                cv2.putText(frame_with_box, "Right face aligned - Press 's' to capture", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if capturing_right:
                    elapsed_time = time.time() - capture_start_time_right
                    seconds_remaining = max(0, 5 - int(elapsed_time))
                    cv2.putText(frame_with_box, f"Capturing right: {seconds_remaining}s", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame_with_box, f"Collected: {collected_front + collected_left + collected_right}", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    if elapsed_time <= 5:
                        facenet_embeddings.append(facenet_embedding)
                        collected_right += 1
                    elif elapsed_time > 5:
                        break  # Exit after right capture completes
            else:
                cv2.putText(frame_with_box, "Turn face right to align with box", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                capturing_right = False
        
        cv2.imshow("Registration", frame_with_box)
        
        key = cv2.waitKey(100) & 0xFF
        if key == ord('s') and aligned and facenet_embedding is not None:
            if phase == "front" and not capturing_front:
                capturing_front = True
                capture_start_time_front = time.time()
            elif phase == "left" and not capturing_left:
                capturing_left = True
                capture_start_time_left = time.time()
            elif phase == "right" and not capturing_right:
                capturing_right = True
                capture_start_time_right = time.time()
        elif key == ord('q'):
            print("Registration cancelled.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    total_collected = collected_front + (collected_left if 'collected_left' in locals() else 0) + (collected_right if 'collected_right' in locals() else 0)
    if total_collected < 5:
        print(f"Insufficient samples collected ({total_collected}). Minimum 5 required. Registration failed.")
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
    
    print(f"User '{user_name}' registered successfully with {total_collected} samples (Front: {collected_front}, Left: {collected_left if 'collected_left' in locals() else 0}, Right: {collected_right if 'collected_right' in locals() else 0}).")
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
            print("✓ YuNet detector loaded successfully")
        else:
            print("✗ YuNet detector failed to load")
            return False
    except Exception as e:
        print(f"✗ YuNet detector error: {str(e)}")
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
    print("\nInitializing system...")
    if not run_diagnostics():
        print("\nSystem diagnostics failed. Please check the errors above.")
        return
    
    while True:
        print("\n=== Face Recognition and Data Collection System ===")
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
            print("Thank you for using the Face Recognition and Data Collection System")
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