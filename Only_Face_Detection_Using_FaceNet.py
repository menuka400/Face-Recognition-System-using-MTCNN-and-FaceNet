import pyrealsense2 as rs
import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d  # Optional for 3D visualization

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

# === Face Detection with BlazeFace and 3D Mapping ===
def detect_3d_face(depth_frame, depth_image, color_image, points, face_detection):
    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)
    
    color_image_with_box = color_image.copy()
    
    if not results.detections:
        cv2.putText(color_image_with_box, "No face detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return None, None, color_image_with_box
    
    # Process the first detected face
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    
    h, w, _ = color_image.shape
    x_min = max(0, int(bbox.xmin * w))
    x_max = min(w, int((bbox.xmin + bbox.width) * w))
    y_min = max(0, int(bbox.ymin * h))
    y_max = min(h, int((bbox.ymin + bbox.height) * h))
    
    # Check if the bounding box is valid
    if x_max <= x_min or y_max <= y_min:
        cv2.putText(color_image_with_box, "Invalid face region", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return None, None, color_image_with_box
    
    # Draw bounding box on RGB image
    cv2.rectangle(color_image_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(color_image_with_box, "Face detected", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Map to 3D using depth data
    depth_roi = depth_image[y_min:y_max, x_min:x_max]
    if depth_roi.size == 0:  # Check if ROI is empty
        return None, None, color_image_with_box
    
    depth_cleaned = cv2.medianBlur(depth_roi, 5)  # Reduce noise
    
    # Extract 3D points within the face region
    vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    height, width = depth_image.shape
    
    face_points = []
    for v in range(height * width):
        row, col = divmod(v, width)
        if (y_min <= row < y_max) and (x_min <= col < x_max):
            depth_value = depth_frame.get_distance(col, row)
            if depth_value > 0:  # Valid depth
                point_3d = rs.rs2_deproject_pixel_to_point(
                    depth_frame.profile.as_video_stream_profile().intrinsics,
                    [col, row], depth_value
                )
                face_points.append(point_3d)
    
    return np.array(face_points), depth_cleaned, color_image_with_box

# === Visualization (Optional with Open3D) ===
def visualize_3d_face(face_points, vis=None):
    if face_points is None or len(face_points) == 0:
        return vis
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(face_points)
    
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Face Detection")
        vis.add_geometry(pcd)
    else:
        vis.clear_geometries()
        vis.add_geometry(pcd)
    
    vis.poll_events()
    vis.update_renderer()
    return vis

# === Continuous Detection Loop ===
def run_3d_face_detection():
    # Initialize MediaPipe BlazeFace
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,  # 0 for short-range model (faces within 2m), 1 for full-range
        min_detection_confidence=0.5
    )
    
    pipeline = initialize_camera()
    vis = None  # Open3D visualizer (optional)
    depth_window_open = False  # Track if Depth ROI window is active
    
    try:
        while True:
            # Capture fresh frames
            depth_frame, depth_image, color_image, ir_image, points = capture_3d_data(pipeline)
            
            if depth_frame is None:
                print("Failed to capture frames - retrying...")
                continue
            
            # Detect face with BlazeFace and map to 3D
            face_points, depth_roi, color_image_with_box = detect_3d_face(
                depth_frame, depth_image, color_image, points, face_detection
            )
            
            # Display RGB with detection status
            cv2.imshow("RGB with BlazeFace", color_image_with_box)
            
            # Manage Depth ROI window
            if depth_roi is not None:
                depth_display = cv2.normalize(depth_roi, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imshow("Depth ROI", depth_display.astype(np.uint8))
                depth_window_open = True
            elif depth_window_open:
                cv2.destroyWindow("Depth ROI")
                depth_window_open = False
            
            # Visualize 3D points (optional, requires Open3D)
            vis = visualize_3d_face(face_points, vis)
            
            # Exit on 'q' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Stopping detection...")
                break
    
    finally:
        face_detection.close()
        pipeline.stop()
        cv2.destroyAllWindows()
        if vis is not None:
            vis.destroy_window()

# === Run the System ===
if __name__ == "__main__":
    print("Starting Continuous 3D Face Detection System with BlazeFace...")
    run_3d_face_detection()