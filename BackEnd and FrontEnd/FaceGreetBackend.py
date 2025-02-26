import cv2
import numpy as np
import sqlite3
import os
import time
from tensorflow.keras import backend as K
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import logging
import websocket
import json

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# WebSocket connection
# uri = "ws://localhost:12393/client-ws"
# try:
#     ws = websocket.create_connection(uri)
#     print("Connected to WebSocket server")
# except Exception as e:
#     print(f"Failed to connect to WebSocket server: {e}")
#     exit(1)

def send_websocket_message(text, type="text-input"):
    return
    """Send a message via WebSocket"""
    try:
        message = {
            "type": type,
            "text": text,
            "images": []
        }
        ws.send(json.dumps(message))
        print(f"Sent WebSocket message: {text}")
    except Exception as e:
        print(f"Error sending WebSocket message: {e}")

# === Model Initialization ===
def load_facenet():
    try:
        facenet = FaceNet()
        return facenet
    except Exception as e:
        print(f"Error loading FaceNet model: {str(e)}")
        return None

def initialize_detector():
    try:
        detector = MTCNN()
        return detector
    except Exception as e:
        print(f"Error initializing MTCNN detector: {str(e)}")
        return None

# === Face Embedding Generation ===
def get_face_embedding(model, face_img):
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
        return None, None, None, frame_with_box
    
    face = faces[0]  # Only process the first face
    if face['confidence'] < 0.9:
        return None, None, None, frame_with_box
    
    x, y, w, h = face['box']
    x_min, y_min = max(0, x), max(0, y)
    x_max, y_max = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    
    face_roi = frame[y_min:y_max, x_min:x_max]
    facenet_embedding = get_face_embedding(facenet_model, face_roi)
    
    cv2.rectangle(frame_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
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

# === Face Recognition Backend ===
def run_face_recognition_backend():
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
    
    print("Starting face recognition backend...")
    
    facenet_threshold = 0.75
    last_recognized_name = None
    greeting_printed = False
    no_face_count = 0
    last_state = None

    while True:
        frame = capture_frame(cap)
        if frame is None:
            continue
        
        start_time = time.time()
        facenet_embedding, _, _, frame_with_box = detect_and_extract_features(frame, detector, facenet_model)
        elapsed_time = time.time() - start_time
        elapsed_ms = int(elapsed_time * 1000)
        elapsed_s = int(elapsed_time) if elapsed_time >= 1 else 0
        progress_output = f"1/1 ━━━━━━━━━━━━━━━━━━━━ {elapsed_s}s {elapsed_s}s/step" if elapsed_s > 0 else f"1/1 ━━━━━━━━━━━━━━━━━━━━ 0s {elapsed_ms}ms/step"
        print(progress_output)
        
        if facenet_embedding is not None:
            best_match, highest_confidence = None, -1
            for user_id, stored_embedding in stored_templates.items():
                similarity = np.dot(facenet_embedding, stored_embedding)
                if similarity > highest_confidence:
                    highest_confidence = similarity
                    best_match = user_id
            
            if highest_confidence > facenet_threshold and not greeting_printed:
                send_websocket_message(f"Hello {best_match}, how's your day?")
                last_recognized_name = best_match
                greeting_printed = True
                no_face_count = 0
                last_state = 'recognized'
            
            elif (highest_confidence > facenet_threshold and 
                  last_recognized_name == best_match and 
                  no_face_count >= 5 and 
                  last_state == 'no_face'):
                send_websocket_message(f"{best_match}, welcome back I miss you")
                no_face_count = 0
                last_state = 'recognized'
        
        else:
            if last_recognized_name is not None and greeting_printed:
                no_face_count += 1
                if no_face_count >= 5 and last_state == 'recognized':
                    send_websocket_message(f"{last_recognized_name}, where you go come back I need you!!!!")
                    last_state = 'no_face'
            elif last_recognized_name is not None and greeting_printed and last_state == 'no_face':
                no_face_count += 1

        cv2.imshow("Face Recognition Backend", frame_with_box)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    ws.close()

# === Program Entry Point ===
if __name__ == "__main__":
    create_database()
    run_face_recognition_backend()