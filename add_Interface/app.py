import cv2
import numpy as np
import sqlite3
import os
import time
from flask import Flask, request, jsonify, render_template
from tensorflow.keras import backend as K
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import logging
import threading

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__)

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

# === Face Detection and Feature Extraction ===
def detect_and_extract_features(frame, detector, facenet_model):
    if frame is None:
        return None, None, None, None
    
    frame_with_box = frame.copy()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_image)
    
    if not faces:
        return None, None, None, frame_with_box
    
    face = faces[0]
    if face['confidence'] < 0.9:
        return None, None, None, frame_with_box
    
    x, y, w, h = face['box']
    x_min, y_min = max(0, x), max(0, y)
    x_max, y_max = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    
    face_roi = frame[y_min:y_max, x_min:x_max]
    facenet_embedding = get_face_embedding(facenet_model, face_roi)
    
    cv2.rectangle(frame_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
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

def clear_database():
    if os.path.exists('face_database.db'):
        os.remove('face_database.db')
        create_database()  # Recreate an empty database
        return True, "Database cleared successfully."
    return False, "No database found to clear."

# === User Registration ===
def register_user(user_name):
    create_database()
    detector = initialize_detector()
    facenet_model = load_facenet()
    if detector is None or facenet_model is None:
        return False, "Model initialization failed"
    
    cap = initialize_camera()
    facenet_embeddings = {'front': [], 'left': [], 'right': []}
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x, center_y = frame_width // 2, frame_height // 2
    oval_width, oval_height = 300, 420
    
    stages = ['front', 'left', 'right']
    stage_instructions = {
        'front': "Align your FRONT face to the oval. Press 's' to start capturing (5 seconds).",
        'left': "Turn your head LEFT to the oval. Press 's' to start capturing (5 seconds).",
        'right': "Turn your head RIGHT to the oval. Press 's' to start capturing (5 seconds)."
    }
    
    for stage in stages:
        collected = 0
        capture_start_time = None
        print(f"\n{stage_instructions[stage]}")
        
        while True:
            frame = capture_frame(cap)
            if frame is None:
                continue
            
            frame_with_box = frame.copy()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_image)
            
            cv2.ellipse(frame_with_box, (center_x, center_y), (oval_width // 2, oval_height // 2),
                        0, 0, 360, (0, 255, 0), 2)
            cv2.putText(frame_with_box, f"Stage: {stage.capitalize()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(frame_with_box, "Press 's' to start", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame_with_box, f"Samples: {collected}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            if stage == 'left':
                cv2.arrowedLine(frame_with_box, (center_x - 60, center_y + oval_height // 2 + 60),
                                (center_x - 180, center_y + oval_height // 2 + 60), (0, 255, 0), 3, tipLength=0.3)
            elif stage == 'right':
                cv2.arrowedLine(frame_with_box, (center_x + 60, center_y + oval_height // 2 + 60),
                                (center_x + 180, center_y + oval_height // 2 + 60), (0, 255, 0), 3, tipLength=0.3)
            
            key = cv2.waitKey(100) & 0xFF
            if key == ord('s') and capture_start_time is None:
                capture_start_time = time.time()
                print(f"Capturing {stage} face for 5 seconds...")
            
            if capture_start_time is not None:
                elapsed_time = time.time() - capture_start_time
                if elapsed_time <= 5.0:
                    if faces and faces[0]['confidence'] >= 0.9:
                        face = faces[0]
                        x, y, w, h = face['box']
                        x_min, y_min = max(0, x), max(0, y)
                        x_max, y_max = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                        face_roi = frame[y_min:y_max, x_min:x_max]
                        facenet_embedding = get_face_embedding(facenet_model, face_roi)
                        
                        face_center_x = x_min + w // 2
                        face_center_y = y_min + h // 2
                        if (abs(face_center_x - center_x) < oval_width // 2 and 
                            abs(face_center_y - center_y) < oval_height // 2):
                            if facenet_embedding is not None:
                                facenet_embeddings[stage].append(facenet_embedding)
                                collected += 1
                                cv2.putText(frame_with_box, "Capturing...", (10, 150),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame_with_box, "Align face to oval", (10, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame_with_box, "No face detected", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    print(f"Finished capturing {stage} face. Collected {collected} samples.")
                    break
            
            cv2.imshow("Registration", frame_with_box)
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False, "Registration interrupted by user"
    
    cap.release()
    cv2.destroyAllWindows()
    
    total_samples = sum(len(facenet_embeddings[stage]) for stage in stages)
    if total_samples < 5:
        return False, f"Insufficient samples collected ({total_samples}). Minimum 5 required."
    
    all_embeddings = np.concatenate([facenet_embeddings[stage] for stage in stages if facenet_embeddings[stage]])
    mean_facenet_embedding = np.mean(all_embeddings, axis=0)
    
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO users (id, facenet_embedding) VALUES (?, ?)',
                   (user_name, mean_facenet_embedding.tobytes()))
    conn.commit()
    conn.close()
    
    return True, f"User '{user_name}' registered with {total_samples} samples (Front: {len(facenet_embeddings['front'])}, Left: {len(facenet_embeddings['left'])}, Right: {len(facenet_embeddings['right'])})."

# === Face Recognition ===
def recognize_face():
    detector = initialize_detector()
    facenet_model = load_facenet()
    if detector is None or facenet_model is None:
        return "Model initialization failed"
    
    cap = initialize_camera()
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, facenet_embedding FROM users')
    stored_templates = {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in cursor.fetchall()}
    conn.close()
    
    if not stored_templates:
        cap.release()
        return "No users registered in the database."
    
    facenet_threshold = 0.75
    
    while True:
        frame = capture_frame(cap)
        if frame is None:
            continue
        
        facenet_embedding, _, _, frame_with_box = detect_and_extract_features(frame, detector, facenet_model)
        
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
            cv2.putText(frame_with_box, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        cv2.imshow("Recognition", frame_with_box)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return "Face recognition stopped."

# === View Registered Users ===
def view_registered_users():
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, registration_date FROM users')
    users = cursor.fetchall()
    conn.close()
    
    if not users:
        return {'users': [], 'message': "No users registered."}
    user_list = [{'id': user_id, 'registration_date': reg_date} for user_id, reg_date in users]
    return {'users': user_list, 'message': f"{len(users)} users registered."}

# === System Diagnostics ===
def run_diagnostics():
    diagnostics = []
    
    try:
        cap = initialize_camera()
        ret, _ = cap.read()
        diagnostics.append("✓ Camera initialization successful" if ret else "✗ Camera not working properly")
        cap.release()
    except Exception as e:
        diagnostics.append(f"✗ Camera error: {str(e)}")
    
    try:
        model = load_facenet()
        diagnostics.append("✓ FaceNet model loaded successfully" if model is not None else "✗ FaceNet model failed to load")
    except Exception as e:
        diagnostics.append(f"✗ FaceNet model error: {str(e)}")
    
    try:
        detector = initialize_detector()
        diagnostics.append("✓ MTCNN detector loaded successfully" if detector is not None else "✗ MTCNN detector failed to load")
    except Exception as e:
        diagnostics.append(f"✗ MTCNN detector error: {str(e)}")
    
    try:
        create_database()
        conn = sqlite3.connect('face_database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users')
        user_count = cursor.fetchone()[0]
        conn.close()
        diagnostics.append(f"✓ Database check successful ({user_count} users registered)")
    except Exception as e:
        diagnostics.append(f"✗ Database error: {str(e)}")
    
    return "\n".join(diagnostics)

# === Flask Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recognition', methods=['GET'])
def start_recognition_route():
    def run_recognition():
        result = recognize_face()
        print(result)
    
    thread = threading.Thread(target=run_recognition)
    thread.start()
    return jsonify({'message': 'Face recognition started. Check the camera window.'})

@app.route('/register_user', methods=['POST'])
def register_user_route():
    data = request.get_json()
    username = data.get('username')
    if not username:
        return jsonify({'message': 'Username is required.'}), 400
    
    success, message = register_user(username)
    return jsonify({'message': message, 'success': success})

@app.route('/view_users', methods=['GET'])
def view_users_route():
    result = view_registered_users()
    return jsonify({'users': result['users'], 'message': result['message']})

@app.route('/clear_database', methods=['POST'])
def clear_database_route():
    success, message = clear_database()
    return jsonify({'message': message, 'success': success})

@app.route('/run_diagnostics', methods=['GET'])
def run_diagnostics_route():
    result = run_diagnostics()
    return jsonify({'message': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)