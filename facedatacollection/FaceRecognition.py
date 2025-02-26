import cv2
import numpy as np
import sqlite3
import os  # Added this import
import time
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
    target_frame_time = 1.0  # 1 second per frame

    while True:
        frame_start_time = time.time()  # Start time of frame processing
        
        frame = capture_frame(cap)
        if frame is None:
            continue
        
        # Process frame
        facenet_embedding, face_roi, bbox, frame_with_box = detect_and_extract_features(
            frame, detector, facenet_model
        )
        
        # Calculate processing time for TensorFlow-style output
        processing_end_time = time.time()
        elapsed_time = processing_end_time - frame_start_time
        elapsed_ms = int(elapsed_time * 1000)
        elapsed_s = int(elapsed_time) if elapsed_time >= 1 else 0
        progress_output = f"1/1 ━━━━━━━━━━━━━━━━━━━━ {elapsed_s}s {elapsed_s}s/step" if elapsed_s > 0 else f"1/1 ━━━━━━━━━━━━━━━━━━━━ 0s {elapsed_ms}ms/step"
        print(progress_output)
        
        # Enforce 1 frame per second
        time_to_sleep = target_frame_time - (time.time() - frame_start_time)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        
        # Calculate FPS based on total frame time (processing + sleep)
        total_frame_time = time.time() - frame_start_time
        fps = 1 / total_frame_time if total_frame_time > 0 else 0
        
        # Display FPS on the frame
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

# === Program Entry Point ===
if __name__ == "__main__":
    try:
        recognize_face()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Shutting down...")
    except Exception as e:
        print(f"\nUnexpected error occurred: {str(e)}")
        print("Please restart the program or contact support.")
    finally:
        cv2.destroyAllWindows()