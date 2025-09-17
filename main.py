#import warnings
#warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

import cv2
import face_recognition
import numpy as np
import os
import json
import pickle
from datetime import datetime
import time
import dlib


class VisitorManager:
    """Handles visitor database operations and tracking"""
    
    def __init__(self, database_file="data/visitor_database.json"):
        self.database_file = database_file
        self.visitors = self.load_database()
    
    def load_database(self):
        """Load visitor database from JSON file"""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'r') as f:
                    return json.load(f)
            except:
                print("⚠️ Could not load visitor database, creating new one")
        return {}
    
    def save_database(self):
        """Save visitor database to JSON file"""
        os.makedirs(os.path.dirname(self.database_file), exist_ok=True)
        with open(self.database_file, 'w') as f:
            json.dump(self.visitors, f, indent=4, default=str)
    
    def add_visitor(self, name):
        """Add a new visitor to the database"""
        if name not in self.visitors:
            self.visitors[name] = {
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'visit_count': 0,
                'visits': []
            }
            self.save_database()
            print(f"➕ Added new visitor: {name}")
        return self.visitors[name]
    
    def record_visit(self, name):
        """Record a visit for a visitor"""
        if name not in self.visitors:
            self.add_visitor(name)
        
        current_time = datetime.now()
        visitor = self.visitors[name]
        
        # Check if this is a recent visit (within 30 seconds) to avoid duplicates
        if visitor['visits']:
            last_visit = datetime.fromisoformat(visitor['visits'][-1])
            if (current_time - last_visit).total_seconds() < 20:
                return visitor['visit_count']
        
        # Record new visit
        visitor['visit_count'] += 1
        visitor['last_seen'] = current_time.isoformat()
        visitor['visits'].append(current_time.isoformat())
        
        self.save_database()
        return visitor['visit_count']
    
    def get_visit_count(self, name):
        """Get visit count for a visitor"""
        if name in self.visitors:
            return self.visitors[name]['visit_count']
        return 0
    
    def print_report(self):
        """Print comprehensive visitor report"""
        print("\n" + "="*50)
        print("📊 VISITOR REPORT")
        print("="*50)
        
        if not self.visitors:
            print("📭 No visitors recorded yet")
            return
        
        # Sort by visit count
        sorted_visitors = sorted(
            self.visitors.items(), 
            key=lambda x: x[1]['visit_count'], 
            reverse=True
        )
        
        total_visitors = len(self.visitors)
        total_visits = sum(v['visit_count'] for v in self.visitors.values())
        
        print(f"👥 Total Unique Visitors: {total_visitors}")
        print(f"🔄 Total Visits: {total_visits}")
        print("-" * 50)
        
        for name, info in sorted_visitors:
            first_seen = datetime.fromisoformat(info['first_seen']).strftime("%Y-%m-%d %H:%M")
            last_seen = datetime.fromisoformat(info['last_seen']).strftime("%Y-%m-%d %H:%M")
            
            print(f"👤 {name}")
            print(f"   📅 First Visit: {first_seen}")
            print(f"   🕐 Last Visit: {last_seen}")
            print(f"   🔢 Visit Count: {info['visit_count']}")
            print("-" * 30)
        
        print("="*50 + "\n")

class FaceRecognitionSystem:
    """Main face recognition and visitor tracking system"""
    
    def __init__(self):
        """Initialize the system"""
        self.visitor_manager = VisitorManager()
        self.known_face_encodings = []
        self.known_face_names = []
        self.setup_system()
        
    def setup_system(self):
        """Setup directories and load existing data"""
        self.setup_directories()
        self.load_known_faces()
        print("✅ System initialized successfully!")
        print(f"📊 Loaded {len(self.known_face_names)} known faces")

        gpu_status = "GPU-Accelerated" if dlib.DLIB_USE_CUDA else "CPU Only"
        print(f"✅ System initialized successfully! ({gpu_status})")
        print(f"📊 Loaded {len(self.known_face_names)} known faces")
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            "data",
            "data/known_faces", 
            "data/captures",
            "logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("📁 Created directory structure")
        
    def load_known_faces(self):
        """Load known faces from directory and create encodings"""
        known_faces_dir = "data/known_faces"
        encodings_file = "data/face_encodings.pkl"
        
        # Try to load cached encodings first
        if os.path.exists(encodings_file):
            try:
                with open(encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"📂 Loaded {len(self.known_face_names)} faces from cache")
                return
            except:
                print("⚠️ Could not load cached encodings, rebuilding...")
        
        # Build encodings from images
        if os.path.exists(known_faces_dir):
            for filename in os.listdir(known_faces_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    name = os.path.splitext(filename)[0]
                    image_path = os.path.join(known_faces_dir, filename)
                    
                    try:
                        # Load and encode face
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        
                        if encodings:
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_names.append(name)
                            print(f"✅ Loaded face: {name}")
                        else:
                            print(f"⚠️ No face found in {filename}")
                    except Exception as e:
                        print(f"❌ Error loading {filename}: {e}")
        
        # Save encodings for faster loading next time
        self.save_encodings()
    
    def save_encodings(self):
        """Save face encodings to cache file"""
        if self.known_face_encodings:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            with open("data/face_encodings.pkl", 'wb') as f:
                pickle.dump(data, f)
            print("💾 Face encodings cached for faster loading")
    
    def add_new_visitor(self, face_image, face_location, name=None):
        """Add a new visitor to the system"""
        if name is None:
            # Generate automatic name based on timestamp
            name = f"Visitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract face from image
        top, right, bottom, left = face_location
        face_crop = face_image[top:bottom, left:right]
        
        # Save face image
        face_filename = f"{name}.jpg"
        face_path = os.path.join("data/known_faces", face_filename)
        cv2.imwrite(face_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
        
        # Get face encoding
        face_encodings = face_recognition.face_encodings(face_image, [face_location])
        if face_encodings:
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            
            # Save updated encodings
            self.save_encodings()
            
            # Add to visitor database
            self.visitor_manager.add_visitor(name)
            
            print(f"✅ New visitor registered: {name}")
            return name
        
        return None
    
    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Log GPU usage status
        if not hasattr(self, '_gpu_status_shown'):
            import dlib
            print(f"🖥️ GPU Acceleration Status:")
            print(f"   • CUDA Enabled: {dlib.DLIB_USE_CUDA}")
            if hasattr(dlib, 'cuda'):
                print(f"   • GPU Devices: {dlib.cuda.get_num_devices()}")
            self._gpu_status_shown = True
        
        # Use CNN model for better accuracy (slower but more confident)
        face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
    
    # Higher quality encodings
        face_encodings = face_recognition.face_encodings(
            rgb_frame, 
            face_locations, 
            num_jitters=3  # More accurate
        )
    
        recognized_faces = []
    
        for (face_encoding, face_location) in zip(face_encodings, face_locations):
        # Stricter tolerance
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding, 
                tolerance=0.5  # More strict
           )
        
            name = "Unknown"
            confidence = 0
        
            if True in matches:
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
               )
                best_match_index = np.argmin(face_distances)
            
                if matches[best_match_index]:
                    confidence = 1 - face_distances[best_match_index]
                
                # Confidence threshold - only accept high confidence
                    if confidence >= 0.5:
                        name = self.known_face_names[best_match_index]
                        visit_count = self.visitor_manager.record_visit(name)
                        print(f"👋 {name} - Visit #{visit_count} (Confidence: {confidence:.2f})")
                    else:
                        print(f"❌ Low confidence match rejected ({confidence:.2f})")
        
            recognized_faces.append({
                'name': name,
                'location': face_location,
                'confidence': confidence
            })
    
        return recognized_faces

    
    def draw_results(self, frame, recognized_faces):
        """Draw bounding boxes and labels"""
        for face_info in recognized_faces:
            name = face_info['name']
            top, right, bottom, left = face_info['location']
            confidence = face_info['confidence']
            
            # Choose colors
            if name == "Unknown":
                color = (0, 0, 255)  # Red
                text_color = (255, 255, 255)
                display_name = "NEW VISITOR"
            else:
                color = (0, 255, 0)  # Green
                text_color = (255, 255, 255)
                visit_count = self.visitor_manager.get_visit_count(name)
                display_name = f"{name} (Visit #{visit_count})"
            
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw text
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, display_name, (left + 6, bottom - 6), font, 0.6, text_color, 1)
            
            # Draw confidence for known faces
            if confidence > 0:
                conf_text = f"Conf: {confidence:.2f}"
                cv2.putText(frame, conf_text, (left + 6, top - 6), font, 0.4, color, 1)
        
        return frame
    
    def run(self):
        """Main application loop"""
        print("🚀 Starting Face Recognition Visitor System...")
        print("📹 Opening camera...")
        
        # Initialize camera
        print("[DEBUG] Attempting to open camera...")
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("❌ Error: Could not open camera")
            print("💡 Troubleshooting:")
            print("   - Check camera permissions")
            print("   - Make sure no other app is using the camera")
            print("   - Try different camera index (1, 2, etc.)")
            
            # Try alternative camera indices
            print("\n[DEBUG] Trying alternative camera indices...")
            for idx in [1, 2, -1]:
                print(f"[DEBUG] Attempting camera index {idx}...")
                video_capture = cv2.VideoCapture(idx)
                if video_capture.isOpened():
                    print(f"[DEBUG] Successfully opened camera at index {idx}")
                    break
                else:
                    print(f"[DEBUG] Failed to open camera at index {idx}")
            
            if not video_capture.isOpened():
                print("[DEBUG] All camera attempts failed")
                return
        
        # Set camera properties for better performance
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera ready!")
        print("📋 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save current frame")
        print("   - Press 'r' to show visitor report")
        print("   - Press 'n' to register unknown person as new visitor")
        print("   - Press 'h' for help")
        
        frame_count = 0
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    break
                
                # Process every 3rd frame for better performance
                if frame_count % 3 == 0:
                    recognized_faces = self.process_frame(frame)
                else:
                    # Use previous results
                    recognized_faces = getattr(self, '_last_recognized_faces', [])
                
                frame = self.draw_results(frame, recognized_faces)
                self._last_recognized_faces = recognized_faces
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 30:
                    fps_end_time = time.time()
                    current_fps = fps_counter / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    fps_counter = 0
                
                # Draw FPS and instructions
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit, 'h' for help", (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.namedWindow('Face Recognition Visitor System', cv2.WINDOW_NORMAL)
                cv2.imshow('Face Recognition Visitor System', frame)
                cv2.moveWindow('Face Recognition Visitor System', 0, 0)  # Move to top-left
                cv2.resizeWindow('Face Recognition Visitor System', 800, 600)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("👋 Shutting down system...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"data/captures/capture_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Frame saved: {filename}")
                elif key == ord('r'):
                    self.visitor_manager.print_report()
                elif key == ord('n'):
                    # Register unknown faces as new visitors
                    for face_info in recognized_faces:
                        if face_info['name'] == "Unknown":
                            # Get current frame in RGB
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            self.add_new_visitor(rgb_frame, face_info['location'])
                            break
                elif key == ord('h'):
                    self.show_help()
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n🛑 System stopped by user (Ctrl+C)")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            # Cleanup
            video_capture.release()
            cv2.destroyAllWindows()
            print("🧹 Camera released and windows closed")
            print("💾 All visitor data saved to database")
    
    def show_help(self):
        """Display help information"""
        print("\n" + "="*50)
        print("📋 HELP - FACE RECOGNITION VISITOR SYSTEM")
        print("="*50)
        print("🎯 PURPOSE:")
        print("   This system tracks visitors using face recognition")
        print("   - Green box = Known visitor (shows visit count)")
        print("   - Red box = New/Unknown visitor")
        print()
        print("⌨️  KEYBOARD CONTROLS:")
        print("   q - Quit the application")
        print("   s - Save current frame as image")
        print("   r - Show detailed visitor report")
        print("   n - Register unknown person as new visitor")
        print("   h - Show this help message")
        print()
        print("📁 FILES & FOLDERS:")
        print("   data/known_faces/     - Add face images here (PersonName.jpg)")
        print("   data/captures/        - Saved screenshots")
        print("   data/visitor_database.json - Visitor records")
        print()
        print("💡 TIPS:")
        print("   - Use good lighting for better recognition")
        print("   - Keep face centered and looking forward")
        print("   - Add multiple photos per person for better accuracy")
        print("="*50 + "\n")

def main():
    """Main entry point"""
    print("🎯 FACE RECOGNITION VISITOR TRACKING SYSTEM")
    print("="*60)
    print("📝 Quick Setup:")
    print("1. Add face images to 'data/known_faces/' folder")
    print("2. Name files as: PersonName.jpg (e.g., John_Smith.jpg)")
    print("3. Run this script and position yourself in front of camera")
    print("4. System will recognize faces and count visits")
    print("="*60)
    
    # Check for required dependencies
    try:
        import cv2
        import face_recognition
        import numpy
        print("✅ All required libraries are installed")
        print(f"[DEBUG] OpenCV version: {cv2.__version__}")
        print("[DEBUG] Starting camera initialization...")
        print("[DEBUG] Checking available cameras...")
    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("🔧 Install requirements:")
        print("   pip install opencv-python face-recognition numpy pillow")
        return
    
    # Check for required dependencies
    try:
        import cv2
        import face_recognition
        import numpy
        print("✅ All required libraries are installed")
    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("🔧 Install requirements:")
        print("   pip install opencv-python face-recognition numpy pillow")
        return
    
    # Check if known_faces directory exists
    if not os.path.exists("data/known_faces"):
        print("⚠️ Creating 'data/known_faces' directory...")
        os.makedirs("data/known_faces", exist_ok=True)
        print("✅ Directory created")
        print("📸 Add face images (JPG/PNG) to data/known_faces/ and restart")
        print("💡 Example: data/known_faces/John_Smith.jpg")
        
        # Create a sample directory structure info
        with open("data/known_faces/README.txt", "w") as f:
            f.write("Add face images here!\n")
            f.write("Name format: FirstName_LastName.jpg\n")
            f.write("Examples:\n")
            f.write("  - John_Smith.jpg\n")
            f.write("  - Alice_Johnson.jpg\n")
            f.write("  - Bob_Wilson.jpg\n")
        
        return
    
    # Check for face images
    known_faces = [f for f in os.listdir("data/known_faces") 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not known_faces:
        print("⚠️ No face images found in 'data/known_faces/'")
        print("📸 Add face images and restart the system")
        print("💡 Supported formats: JPG, JPEG, PNG")
        print("💡 Naming: PersonName.jpg (e.g., John_Smith.jpg)")
        return
    
    print(f"📊 Found {len(known_faces)} face image(s):")
    for face in known_faces:
        print(f"   👤 {os.path.splitext(face)[0]}")
    
    print("\n🚀 Starting system...")
    
    try:
        system = FaceRecognitionSystem()
        system.run()
    except Exception as e:
        print(f"❌ System error: {e}")
        print("💡 Try restarting the application")

if __name__ == "__main__":
    main()
