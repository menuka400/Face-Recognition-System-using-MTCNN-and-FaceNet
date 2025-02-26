import cv2
import numpy as np
from PIL import Image
import psycopg2
import os
from tabulate import tabulate
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import time
import websocket
import json


current_detected_person="Unknown"
current_detected_person_count=0

recognized_person="Unknown"
after_recognized_count=0




uri = "ws://localhost:12393/client-ws"
ws = websocket.create_connection(uri)


def connect_and_send(text="",type="text-input"):
    # return
    print('msg sending')
    
    try:
        message = {
            "type": type,
            "text": text,
            "images": []
        }
        
        ws.send(json.dumps(message))
        print("Message sent to WebSocket server")
        # ws.close()
    except Exception as e:
        print(f"Error: {e}")




def connect_db():
    return psycopg2.connect("postgres://avnadmin:AVNS_F9QTOJet-vzalfnV2-T@pg-9c3cfe6-menuka-1.h.aivencloud.com:21721/defaultdb?sslmode=require")

def initialize_database():
    """Create the required table if it doesn't exist and handle migration if needed"""
    conn = connect_db()
    cur = conn.cursor()
    try:
        # First, check if we need to clear existing records due to embedding size change
        cur.execute("""
            SELECT embedding FROM face_records LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            existing_embedding = row[0]
            if len(existing_embedding) != 512:  # FaceNet embedding size
                print("Detected incompatible embedding format in database. Clearing existing records...")
                cur.execute("DROP TABLE face_records")
                conn.commit()
                print("Database cleared successfully.")

        # Create the table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_records (
                id VARCHAR(255) PRIMARY KEY,
                embedding FLOAT[] NOT NULL
            )
        """)
        conn.commit()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
    finally:
        cur.close()
        conn.close()

class FaceProcessor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=20, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )
        
        # Initialize FaceNet model
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.resnet = self.resnet.to(self.device)
        
        # Load registered faces
        self.registered_faces = self.load_registered_faces()

    def load_registered_faces(self):
        """Load all registered faces from database"""
        registered_faces = []
        conn = connect_db()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, embedding FROM face_records")
            registered_faces = cur.fetchall()
        except Exception as e:
            print(f"Error loading registered faces: {str(e)}")
        finally:
            cur.close()
            conn.close()
        return registered_faces

    def refresh_registered_faces(self):
        """Refresh the registered faces list"""
        self.registered_faces = self.load_registered_faces()

    def get_face_embedding(self, input_data):
        """Get face embedding from either tensor or image"""
        try:
            if isinstance(input_data, torch.Tensor):
                # Input is already a face tensor from MTCNN
                face_tensor = input_data.to(self.device)
            else:
                # Input is an image
                if isinstance(input_data, np.ndarray):
                    input_data = Image.fromarray(cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB))
                
                # Use MTCNN to get face tensor
                face_tensor = self.mtcnn(input_data)
                if face_tensor is None:
                    return None
                face_tensor = face_tensor.to(self.device)
            
            # Ensure 4D tensor [batch_size, channels, height, width]
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.resnet(face_tensor)
            
            return embedding.cpu().numpy()[0]
        except Exception as e:
            print(f"Error processing face: {str(e)}")
            return None

    def recognize_face_realtime(self, frame):
        """Process a single frame and return recognition results"""
        try:
            # Convert to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_frame)
            
            # Get both boxes and probabilities
            boxes, probs = self.mtcnn.detect(img_pil)
            
            results = []
            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    try:
                        # Only process faces with high confidence
                        if prob < 0.9:  # Increased confidence threshold
                            continue
                            
                        box = box.astype(int)
                        
                        # Add margin to face crop
                        margin = 20
                        h = box[3] - box[1]
                        w = box[2] - box[0]
                        box[0] = max(0, box[0] - margin)
                        box[1] = max(0, box[1] - margin)
                        box[2] = min(frame.shape[1], box[2] + margin)
                        box[3] = min(frame.shape[0], box[3] + margin)
                        
                        # Extract and process face
                        face_region = rgb_frame[box[1]:box[3], box[0]:box[2]]
                        face_pil = Image.fromarray(face_region)
                        
                        # Get aligned face tensor directly from MTCNN
                        face_tensor = self.mtcnn(face_pil)
                        
                        if face_tensor is not None:
                            embedding = self.get_face_embedding(face_tensor)
                            
                            if embedding is not None:
                                # Compare with registered faces
                                best_match = None
                                highest_similarity = 0
                                
                                for user_id, stored_embedding in self.registered_faces:
                                    similarity = calculate_similarity(embedding, stored_embedding)
                                    if similarity > highest_similarity:
                                        highest_similarity = similarity
                                        best_match = user_id
                                
                                results.append({
                                    'box': box,
                                    'user_id': best_match if highest_similarity >= 0.75 else "Unknown",
                                    'confidence': highest_similarity
                                })
                    
                    except Exception as e:
                        print(f"Error processing individual face: {str(e)}")
                        continue
            
            return results
        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
            return []

def capture_face(face_processor):
    cap = cv2.VideoCapture(1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)
        
        # Detect face
        boxes, _ = face_processor.mtcnn.detect(img_pil)
        
        if boxes is not None:
            for box in boxes:
                box = box.astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        cv2.imshow("Face Detection - Press 's' to Save, 'q' to Quit", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and boxes is not None:
            save_path = "stored-faces/new_user.jpg"
            cv2.imwrite(save_path, frame)
            cap.release()
            cv2.destroyAllWindows()
            return save_path
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return None

def calculate_similarity(embedding1, embedding2):
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    # Normalize embeddings
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    # Calculate cosine similarity
    similarity = np.dot(emb1, emb2)
    return similarity

def register_face(face_processor):
    user_id = input("Enter ID number: ")
    
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM face_records WHERE id = %s", (user_id,))
        if cur.fetchone():
            print("Error: This ID is already registered!")
            return
        
        # Capture multiple face samples
        embeddings = []
        for i in range(10):
            print(f"\nCapturing face sample {i+1}/10...")
            img_path = capture_face(face_processor)
            if img_path:
                img = Image.open(img_path)
                embedding = face_processor.get_face_embedding(img)
                if embedding is not None:
                    embeddings.append(embedding)
                    print(f"Sample {i+1} captured successfully!")
                else:
                    print("Failed to detect face clearly. Please try again.")
                    continue
            else:
                print("Failed to capture face. Please try again.")
                continue
            
        if len(embeddings) == 0:
            print("Failed to capture any valid face samples. Please try again.")
            return
        
        # Use average embedding
        average_embedding = np.mean(embeddings, axis=0)
        
        cur.execute("INSERT INTO face_records (id, embedding) VALUES (%s, %s)",
                   (user_id, average_embedding.tolist()))
        conn.commit()
        print(f"New user registered successfully with ID: {user_id}!")
        
    except Exception as e:
        print(f"Error during registration: {str(e)}")
    finally:
        cur.close()
        conn.close()

def real_time_recognition(face_processor):
    cap = cv2.VideoCapture(1)
    global current_detected_person,current_detected_person_count,after_recognized_count,recognized_person
    
    # Set up window
    cv2.namedWindow("Real-time Face Recognition", cv2.WINDOW_NORMAL)
    
    last_refresh = time.time()
    refresh_interval = 10  # Refresh registered faces every 10 seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Refresh registered faces periodically
        if time.time() - last_refresh > refresh_interval:
            face_processor.refresh_registered_faces()
            last_refresh = time.time()
        
        # Process frame
        results = face_processor.recognize_face_realtime(frame)

        if(recognized_person != 'Unknown') :
            after_recognized_count+=1
        
        # Draw results
        for result in results:
            box = result['box']
            user_id = result['user_id']
            confidence = result['confidence']
            
            # Draw rectangle around face
            color = (0, 255, 0) if user_id != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Draw text
            text = f"{user_id} ({confidence:.2%})"
            cv2.putText(frame, text, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            

            if (current_detected_person!='Unknown'):
                current_detected_person_count+=1
            else:
                current_detected_person_count=0

            current_detected_person=user_id

            # if(recognized_person != 'Unknown') :
            #     after_recognized_count+=1
            # else:   
            #     pass

            # if(recognized_person != 'Unknown') :
            #     after_recognized_count+=1


            # if(recognize_face!='Unknown'):
            #     after_recognized_count+=1
            # else:
            #     after_recognized_count=0


            #--------------------------- 

            # if(current_detected_person=='Unknown'):
            #     after_recognized_count+=1
            #     if(after_recognized_count>20 and recognized_person!='Unknown'):
            #          print(f"{recognized_person} you are still here  ?? ")
            #          after_recognized_count=0
            #          recognized_person='Unknown'
            # else:
            #     after_recognized_count=0
            

            if (current_detected_person_count>10):
                # print('if logic triggered')
                # print('recognized_person',recognized_person)
                # print('current_detected_person',current_detected_person)

                if(recognized_person == 'Unknown' and current_detected_person!='Unknown'):
                    print(f"Hello {current_detected_person} ........")
                    connect_and_send(f"my name is  {current_detected_person} ")
                    recognized_person=current_detected_person
                elif(recognized_person != 'Unknown' and recognized_person!=current_detected_person):
                    # print(f"Hello {current_detected_person} ........")
                    recognized_person=current_detected_person

                
                # if(recognized_person !='Unknown' and current_detected_person=='Unknown'):
                #     print(f"{recognized_person} you are still here  ?? ")
                    
                    

                # if(recognized_person!=current_detected_person and recognized_person == 'Unknown'):

                #     print(f"Hello {current_detected_person} ........")
                #     recognized_person=current_detected_person
                # after_recognized_count=1
                current_detected_person_count=0
            


            # if(recognize_face=='Unknown'):
            #     after_recognized_count+=1
            #     if(after_recognized_count>150):
            #         # recognized_person='Unknown'
            #         after_recognized_count=0

        # print('recognized_person',recognized_person,after_recognized_count)
        
      
        if(len(results) < 1):
            # if(recognized_person != 'Unknown') :
            #     after_recognized_count+=1

            # if(current_detected_person=='Unknown'):
            # after_recognized_count+=1
            # print(after_recognized_count)
            # print(recognized_person)
            if(after_recognized_count>25 and recognized_person!='Unknown'):
                print(f"{recognized_person} you are still here  ?? ")
                connect_and_send(f"Oh my God {recognized_person} ! Where are gone ? . i can not see your face. please look at me!","text-input-bypass_llm")
                after_recognized_count=0
                recognized_person='Unknown'
            else:
                after_recognized_count=0
            

        # print(len(results))

        # Show frame
        cv2.imshow("Real-time Face Recognition", frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def recognize_face(face_processor):
    img_path = capture_face(face_processor)
    if img_path:
        try:
            img = Image.open(img_path)
            current_embedding = face_processor.get_face_embedding(img)
            
            if current_embedding is None:
                print("Failed to detect face clearly. Please try again.")
                return
            
            conn = connect_db()
            cur = conn.cursor()
            
            cur.execute("SELECT id, embedding FROM face_records")
            rows = cur.fetchall()
            
            if not rows:
                print("No registered users in the database!")
                return
            
            # Enhanced matching logic
            similarities = []
            for row in rows:
                user_id, stored_embedding = row
                try:
                    similarity = calculate_similarity(current_embedding, stored_embedding)
                    similarities.append((user_id, similarity))
                except Exception as e:
                    print(f"Error comparing with user {user_id}: {str(e)}")
                    continue
            
            if not similarities:
                print("Could not compare with any stored faces. Please try again.")
                return
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Adaptive threshold
            top_similarity = similarities[0][1]
            threshold = 0.75  # Base threshold for FaceNet
            
            if top_similarity >= threshold:
                print(f"User recognized as ID: {similarities[0][0]}")
                print(f"Confidence: {top_similarity:.2%}")
                
                # Show top 3 matches
                print("\nTop 3 matches:")
                for user_id, sim in similarities[:min(3, len(similarities))]:
                    print(f"ID: {user_id}, Confidence: {sim:.2%}")
            else:
                print("Access denied: Face not recognized")
                print(f"Best match confidence: {top_similarity:.2%}")
        
        except Exception as e:
            print(f"Error during recognition: {str(e)}")
        finally:
            cur.close()
            conn.close()

def view_users():
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM face_records ORDER BY id")
        rows = cur.fetchall()
        
        if not rows:
            print("\nNo users registered in the database.")
            return
        
        table_data = [[i+1, row[0]] for i, row in enumerate(rows)]
        headers = ["No.", "User ID"]
        
        print("\nRegistered Users:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print(f"\nTotal users registered: {len(rows)}")
        
    except Exception as e:
        print(f"Error viewing users: {str(e)}")
    finally:
        cur.close()
        conn.close()

def remove_user():
    view_users()
    
    user_id = input("\nEnter the ID of the user to remove (or press Enter to cancel): ")
    if not user_id:
        return
    
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM face_records WHERE id = %s", (user_id,))
        if not cur.fetchone():
            print(f"No user found with ID: {user_id}")
            return
        
        confirm = input(f"Are you sure you want to remove user {user_id}? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Deletion cancelled.")
            return
        
        cur.execute("DELETE FROM face_records WHERE id = %s", (user_id,))
        conn.commit()
        print(f"User {user_id} has been removed successfully!")
        
    except Exception as e:
        print(f"Error removing user: {str(e)}")
    finally:
        cur.close()
        conn.close()

def view_and_edit_database():
    while True:
        print("\nDatabase Management Menu:")
        print("1. View all registered users")
        print("2. Remove a user")
        print("3. Return to main menu")
        
        choice = input("Enter your choice: ")
        
        if choice == "1":
            view_users()
        elif choice == "2":
            remove_user()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")

def main():
    print("Initializing face recognition system...")
    os.makedirs("stored-faces", exist_ok=True)
    initialize_database()
    
    # Initialize face processor
    face_processor = FaceProcessor()
    print("System initialized successfully!")
    
    while True:
        print("\nMain Menu:")
        print("1. Start Face Recognition")
        print("2. Start Real-time Face Recognition")  # New option
        print("3. Create New User")
        print("4. View User Database and Edit")
        print("5. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == "1":
            recognize_face(face_processor)
        elif choice == "2":
            real_time_recognition(face_processor)
        elif choice == "3":
            register_face(face_processor)
            face_processor.refresh_registered_faces()  # Refresh after registration
        elif choice == "4":
            view_and_edit_database()
            face_processor.refresh_registered_faces()  # Refresh after potential changes
        elif choice == "5":
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()