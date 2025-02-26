import pyrealsense2 as rs
import cv2
import numpy as np
import mediapipe as mp
import sqlite3
import time
import os

# === Camera Initialization ===
def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
    pipeline.start(config)
    return pipeline

# === Capture 3D Data ===
def capture_3d_data(pipeline):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    ir_frame = frames.get_infrared_frame()
    
    if not (depth_frame and color_frame and ir_frame):
        return None, None, None, None, None
    
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    ir_image = np.asanyarray(ir_frame.get_data())
    
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    return depth_frame, depth_image, color_image, ir_image, points

# === BlazeFace Detection and 3D Extraction ===
def detect_3d_face(depth_frame, depth_image, color_image, points, face_detection):
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)
    
    color_image_with_box = color_image.copy()
    
    if not results.detections:
        cv2.putText(color_image_with_box, "No face detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return None, None, color_image_with_box
    
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    
    h, w, _ = color_image.shape
    x_min = max(0, int(bbox.xmin * w))
    x_max = min(w, int((bbox.xmin + bbox.width) * w))
    y_min = max(0, int(bbox.ymin * h))
    y_max = min(h, int((bbox.ymin + bbox.height) * h))
    
    if x_max <= x_min or y_max <= y_min:
        cv2.putText(color_image_with_box, "Invalid face region", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return None, None, color_image_with_box
    
    cv2.rectangle(color_image_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(color_image_with_box, "Face detected", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    depth_roi = depth_image[y_min:y_max, x_min:x_max]
    if depth_roi.size == 0:
        return None, None, color_image_with_box
    
    depth_cleaned = cv2.medianBlur(depth_roi, 5)
    
    face_points = []
    for row in range(y_min, y_max):
        for col in range(x_min, x_max):
            depth_value = depth_frame.get_distance(col, row)
            if depth_value > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(
                    depth_frame.profile.as_video_stream_profile().intrinsics,
                    [col, row], depth_value
                )
                face_points.append(point_3d)
    
    return np.array(face_points), depth_cleaned, color_image_with_box

# === Feature Extraction with Curvature ===
def extract_features(face_points):
    if face_points is None or len(face_points) < 10:
        return None
    
    mean_xyz = np.mean(face_points, axis=0)
    std_xyz = np.std(face_points, axis=0)
    
    sampled_points = face_points[::max(1, len(face_points) // 100)]
    curvatures = []
    for i in range(1, len(sampled_points) - 1):
        p0, p1, p2 = sampled_points[i-1], sampled_points[i], sampled_points[i+1]
        curvature = abs(2 * p1[2] - p0[2] - p2[2]) / (np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + 1e-6)
        curvatures.append(curvature)
    
    curvature_features = [np.mean(curvatures), np.std(curvatures)] if curvatures else [0.0, 0.0]
    
    return np.concatenate([mean_xyz, std_xyz, curvature_features])

# === User Registration ===
def register_user(user_name, num_samples=15):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    pipeline = initialize_camera()
    templates = []
    
    print(f"Registering user '{user_name}'. Please face the camera...")
    cv2.namedWindow("Registration", cv2.WINDOW_NORMAL)
    
    collected = 0
    while collected < num_samples:
        depth_frame, depth_image, color_image, _, points = capture_3d_data(pipeline)
        if depth_frame is None:
            continue
        
        face_points, _, color_image_with_box = detect_3d_face(depth_frame, depth_image, color_image, points, face_detection)
        
        cv2.imshow("Registration", color_image_with_box)
        if face_points is not None:
            features = extract_features(face_points)
            if features is not None:
                templates.append(features)
                collected += 1
                print(f"Captured {collected}/{num_samples} samples")
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            print("Registration interrupted")
            break
    
    pipeline.stop()
    cv2.destroyAllWindows()
    face_detection.close()
    
    if len(templates) < 5:
        print("Insufficient samples collected. Registration failed.")
        return False
    
    user_template = np.mean(templates, axis=0)
    
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, template BLOB)')
    cursor.execute('INSERT OR REPLACE INTO users (id, template) VALUES (?, ?)', 
                   (user_name, user_template.tobytes()))
    conn.commit()
    conn.close()
    
    print(f"User '{user_name}' registered with {len(templates)} samples.")
    return True

# === Face Recognition ===
def recognize_face():
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    pipeline = initialize_camera()
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, template FROM users')
    stored_templates = {}
    for row in cursor.fetchall():
        user_id, template_data = row
        template = np.frombuffer(template_data, dtype=np.float64)
        if template.shape[0] != 8:  # Check for old 6-feature templates
            print(f"Warning: Template for '{user_id}' has {template.shape[0]} features, expected 8. Please re-register this user.")
            continue
        stored_templates[user_id] = template
    conn.close()
    
    if not stored_templates:
        print("No valid users registered in the database. Please register users with the current feature set.")
        pipeline.stop()
        face_detection.close()
        return
    
    print("Starting face recognition. Press 'q' to stop...")
    cv2.namedWindow("Recognition", cv2.WINDOW_NORMAL)
    
    threshold = 0.5
    
    while True:
        depth_frame, depth_image, color_image, _, points = capture_3d_data(pipeline)
        if depth_frame is None:
            continue
        
        face_points, _, color_image_with_box = detect_3d_face(depth_frame, depth_image, color_image, points, face_detection)
        
        if face_points is not None:
            new_features = extract_features(face_points)
            if new_features is not None:
                min_distance = float('inf')
                recognized_id = None
                
                print("\nDistances to registered users:")
                for user_id, template in stored_templates.items():
                    distance = np.linalg.norm(new_features - template)
                    print(f"{user_id}: {distance:.3f}")
                    if distance < min_distance:
                        min_distance = distance
                        recognized_id = user_id
                
                if min_distance < threshold:
                    cv2.putText(color_image_with_box, f"Recognized: {recognized_id} (dist: {min_distance:.3f})", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(color_image_with_box, f"Unknown face (dist: {min_distance:.3f})", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Recognition", color_image_with_box)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    pipeline.stop()
    cv2.destroyAllWindows()
    face_detection.close()

# === View Registered Users ===
def view_registered_users():
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users')
    users = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    if not users:
        print("No users registered.")
        return
    
    while True:
        print("\nRegistered Users:")
        for i, user in enumerate(users, 1):
            print(f"{i}. {user}")
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
                    user_id = users[user_idx]
                    conn = sqlite3.connect('face_database.db')
                    cursor = conn.cursor()
                    cursor.execute('SELECT template FROM users WHERE id = ?', (user_id,))
                    template_data = cursor.fetchone()[0]
                    template = np.frombuffer(template_data, dtype=np.float64)
                    conn.close()
                    print(f"\nDetails for '{user_id}':")
                    print(f"Template: {template} (Features: {template.shape[0]})")
                else:
                    print("Invalid user number.")
            except ValueError:
                print("Please enter a valid number.")
        
        elif choice == '2':
            user_num = input("Enter user number to remove: ").strip()
            try:
                user_idx = int(user_num) - 1
                if 0 <= user_idx < len(users):
                    user_id = users[user_idx]
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

# === Main Menu ===
def main_menu():
    while True:
        print("\n=== 3D Face Recognition System ===")
        print("1. Start face recognition in real time")
        print("2. Register a user")
        print("3. View registered users")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            recognize_face()
        
        elif choice == '2':
            user_name = input("Enter user name: ").strip()
            if user_name:
                register_user(user_name)
            else:
                print("User name cannot be empty.")
        
        elif choice == '3':
            view_registered_users()
        
        elif choice == '4':
            print("Exiting system...")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

# === Run the System ===
if __name__ == "__main__":
    # Optional: Clear old database to ensure consistency (uncomment if needed)
    # if os.path.exists('face_database.db'):
    #     os.remove('face_database.db')
    #     print("Old database cleared. Please re-register users.")
    
    main_menu()