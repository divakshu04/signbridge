import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from collections import deque, Counter

class SignLanguageDetector:
    def __init__(self, model_path='signbridge_isl_29_signs.h5'):
        # Load your trained model
        self.model = load_model(model_path)
        
        # Same actions list as training
        self.actions = np.array([
            'hello', 'thanks', 'sorry', 'please', 'yes', 'no',
            'I', 'you', 'name', 'time', 'what', 'where', 'how',
            'help', 'learn', 'work', 'eat', 'drink', 'home',
            'good', 'bad', 'happy', 'sad', 'tired',
            'one', 'two', 'three', 'four', 'five'
        ])
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # SLIDING WINDOW for real-time detection
        self.sequence_length = 30
        self.sequence = deque(maxlen=self.sequence_length)  # Auto-removes old frames
        
        # Real-time prediction smoothing
        self.prediction_buffer = deque(maxlen=10)  # Last 10 predictions
        self.current_prediction = ""
        self.confidence = 0.0
        
        # Confidence threshold
        self.threshold = 0.35  # Lower for more responsive detection
        
        # Sentence building - REDUCED for faster response
        self.sentence = []
        self.last_prediction = ""
        self.same_prediction_count = 0
        self.frames_to_confirm = 12  # Reduced from 20 - faster word lock!
        
        # Video capture
        self.cap = None
        self.is_running = False
        
    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe results"""
        keypoints = np.zeros(21 * 3)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            keypoints = np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten()
        return keypoints
    
    def mediapipe_detection(self, image, model):
        """Process frame with MediaPipe"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    
    def draw_styled_landmarks(self, image, results):
        """Draw hand landmarks"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
    
    def predict_sign_realtime(self):
        """Real-time prediction with sliding window"""
        if len(self.sequence) == self.sequence_length:
            # Convert deque to numpy array for prediction
            input_sequence = np.array(list(self.sequence))
            
            # Check if sequence has actual hand data (not all zeros)
            # Check last 10 frames for hand presence (reduced from 15)
            recent_frames = list(self.sequence)[-10:]
            hand_detected_count = sum(1 for frame in recent_frames if np.any(frame))
            
            # Only predict if hand is detected in at least 6 of last 10 frames
            if hand_detected_count >= 6:
                # Make prediction
                res = self.model.predict(np.expand_dims(input_sequence, axis=0), verbose=0)[0]
                predicted_idx = np.argmax(res)
                confidence = res[predicted_idx]
                
                # Only consider predictions above threshold
                if confidence > self.threshold:
                    predicted_sign = self.actions[predicted_idx]
                    
                    # Add to prediction buffer for smoothing
                    self.prediction_buffer.append(predicted_sign)
                    
                    # Use majority voting from last 10 predictions
                    if len(self.prediction_buffer) >= 4:  # Reduced from 5
                        # Get most common prediction
                        prediction_counts = Counter(self.prediction_buffer)
                        most_common_sign, count = prediction_counts.most_common(1)[0]
                        
                        # Update current prediction if it appears frequently enough
                        if count >= 2:  # Reduced from 3 - more responsive!
                            self.current_prediction = most_common_sign
                            self.confidence = confidence
                            
                            # Track for sentence building
                            if most_common_sign == self.last_prediction:
                                self.same_prediction_count += 1
                            else:
                                self.same_prediction_count = 0
                                self.last_prediction = most_common_sign
                            
                            # Add to sentence after consistent detection
                            if self.same_prediction_count == self.frames_to_confirm:
                                if len(self.sentence) == 0 or self.sentence[-1] != most_common_sign:
                                    self.sentence.append(most_common_sign)
                                    self.same_prediction_count = 0  # Reset after adding
            else:
                # No hand detected - clear prediction display
                self.current_prediction = ""
                self.confidence = 0.0
                self.prediction_buffer.clear()
                # Don't reset same_prediction_count here to allow motion completion
        
        return self.current_prediction, self.confidence


class SignLanguageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SignBridge - Real-time Sign Language Detection")
        self.root.geometry("1200x750")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize detector
        self.detector = SignLanguageDetector()
        
        # Setup GUI
        self.setup_gui()
        
        # Threading
        self.running = False
        
    def setup_gui(self):
        # Title
        title_label = tk.Label(
            self.root, 
            text="SignBridge - Real-time Detection",
            font=("Arial", 24, "bold"),
            bg='#2b2b2b',
            fg='#00ff88'
        )
        title_label.pack(pady=10)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Video frame
        video_frame = tk.Frame(main_frame, bg='#1a1a1a', relief=tk.RAISED, borderwidth=2)
        video_frame.pack(side=tk.LEFT, padx=10)
        
        self.video_label = tk.Label(video_frame, bg='#1a1a1a')
        self.video_label.pack(padx=10, pady=10)
        
        # Right panel
        right_panel = tk.Frame(main_frame, bg='#2b2b2b')
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Current prediction display
        pred_frame = tk.Frame(right_panel, bg='#1a1a1a', relief=tk.RAISED, borderwidth=2)
        pred_frame.pack(fill=tk.BOTH, expand=False, pady=10)
        
        tk.Label(
            pred_frame,
            text="Current Sign:",
            font=("Arial", 14, "bold"),
            bg='#1a1a1a',
            fg='#ffffff'
        ).pack(pady=5)
        
        self.prediction_label = tk.Label(
            pred_frame,
            text="---",
            font=("Arial", 36, "bold"),
            bg='#1a1a1a',
            fg='#00ff88',
            wraplength=300,
            height=2
        )
        self.prediction_label.pack(pady=10)
        
        # Confidence bar
        tk.Label(
            pred_frame,
            text="Confidence:",
            font=("Arial", 11),
            bg='#1a1a1a',
            fg='#ffffff'
        ).pack()
        
        self.confidence_var = tk.DoubleVar()
        self.confidence_bar = ttk.Progressbar(
            pred_frame,
            variable=self.confidence_var,
            maximum=100,
            length=250,
            mode='determinate'
        )
        self.confidence_bar.pack(pady=5)
        
        self.confidence_label = tk.Label(
            pred_frame,
            text="0%",
            font=("Arial", 10),
            bg='#1a1a1a',
            fg='#ffffff'
        )
        self.confidence_label.pack(pady=5)
        
        # Sentence display
        sentence_frame = tk.Frame(right_panel, bg='#1a1a1a', relief=tk.RAISED, borderwidth=2)
        sentence_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        sentence_header = tk.Frame(sentence_frame, bg='#1a1a1a')
        sentence_header.pack(fill=tk.X, pady=5)
        
        tk.Label(
            sentence_header,
            text="📝 Detected Sequence:",
            font=("Arial", 14, "bold"),
            bg='#1a1a1a',
            fg='#ffffff'
        ).pack(side=tk.LEFT, padx=10)
        
        clear_btn = tk.Button(
            sentence_header,
            text="Clear",
            command=self.clear_sentence,
            font=("Arial", 10),
            bg='#ff5555',
            fg='#ffffff',
            relief=tk.FLAT,
            cursor="hand2"
        )
        clear_btn.pack(side=tk.RIGHT, padx=10)
        
        # Sentence text with scrollbar
        sentence_scroll_frame = tk.Frame(sentence_frame, bg='#1a1a1a')
        sentence_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        sentence_scrollbar = tk.Scrollbar(sentence_scroll_frame)
        sentence_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.sentence_text = tk.Text(
            sentence_scroll_frame,
            height=8,
            font=("Arial", 14),
            bg='#2b2b2b',
            fg='#ffffff',
            relief=tk.FLAT,
            wrap=tk.WORD,
            yscrollcommand=sentence_scrollbar.set
        )
        self.sentence_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sentence_scrollbar.config(command=self.sentence_text.yview)
        
        # Instructions
        instructions = tk.Text(
            right_panel,
            height=7,
            width=35,
            font=("Arial", 10),
            bg='#1a1a1a',
            fg='#ffffff',
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        instructions.pack(pady=10)
        instructions.insert(1.0, 
            "📌 REAL-TIME MODE:\n\n"
            "✓ Predictions happen continuously\n"
            "✓ Hold signs for ~0.5 seconds\n"
            "✓ Words auto-add quickly!\n"
            "✓ Works with moving signs!\n\n"
            "💡 For dynamic signs:\n"
            "  Repeat motion 2-3 times"
        )
        instructions.config(state=tk.DISABLED)
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg='#2b2b2b')
        button_frame.pack(pady=10)
        
        self.start_button = tk.Button(
            button_frame,
            text="▶ Start Detection",
            command=self.start_detection,
            font=("Arial", 12, "bold"),
            bg='#00ff88',
            fg='#000000',
            width=15,
            height=2,
            relief=tk.RAISED,
            cursor="hand2"
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(
            button_frame,
            text="⏹ Stop Detection",
            command=self.stop_detection,
            font=("Arial", 12, "bold"),
            bg='#ff5555',
            fg='#ffffff',
            width=15,
            height=2,
            relief=tk.RAISED,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
    def clear_sentence(self):
        """Clear the sentence"""
        self.detector.sentence = []
        self.sentence_text.delete(1.0, tk.END)
        self.detector.same_prediction_count = 0
        self.detector.last_prediction = ""
        
    def start_detection(self):
        if not self.running:
            self.running = True
            self.detector.cap = cv2.VideoCapture(0)
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Start detection in separate thread
            detection_thread = threading.Thread(target=self.detection_loop)
            detection_thread.daemon = True
            detection_thread.start()
    
    def stop_detection(self):
        self.running = False
        if self.detector.cap:
            self.detector.cap.release()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.video_label.config(image='')
    
    def detection_loop(self):
        with self.detector.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        ) as hands:
            
            while self.running:
                ret, frame = self.detector.cap.read()
                if not ret:
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detection
                image, results = self.detector.mediapipe_detection(frame, hands)
                self.detector.draw_styled_landmarks(image, results)
                
                # Extract keypoints
                keypoints = self.detector.extract_keypoints(results)
                hand_present = np.any(keypoints)
                
                # Always add to sequence (even zeros for smooth transitions)
                self.detector.sequence.append(keypoints)
                
                # Real-time prediction every frame
                prediction, confidence = self.detector.predict_sign_realtime()
                
                # Update GUI
                self.root.after(0, self.update_prediction, prediction, confidence)
                self.root.after(0, self.update_sentence)
                
                # Visual indicators on frame
                frames_collected = len(self.detector.sequence)
                
                # Hand detection status
                hand_text = "Hand: DETECTED" if hand_present else "Hand: NOT DETECTED"
                hand_color = (0, 255, 0) if hand_present else (0, 0, 255)
                cv2.putText(
                    image,
                    hand_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    hand_color,
                    2
                )
                
                # Current prediction on video
                if self.detector.current_prediction and hand_present:
                    cv2.putText(
                        image,
                        f'{self.detector.current_prediction.upper()}',
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 255),
                        3
                    )
                    
                    # Progress bar for word confirmation (only when hand present)
                    progress = min(100, (self.detector.same_prediction_count / self.detector.frames_to_confirm) * 100)
                    bar_width = 200
                    bar_height = 20
                    cv2.rectangle(image, (10, 150), (10 + bar_width, 150 + bar_height), (50, 50, 50), -1)
                    cv2.rectangle(image, (10, 150), (10 + int(bar_width * progress / 100), 150 + bar_height), (0, 255, 0), -1)
                    cv2.putText(image, 'Word Lock', (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    # Show "waiting" message when no hand
                    if not hand_present:
                        cv2.putText(
                            image,
                            'Show hand to detect sign...',
                            (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (150, 150, 150),
                            2
                        )
                
                # Convert and display
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = image.resize((640, 480))
                photo = ImageTk.PhotoImage(image)
                
                self.root.after(0, self.update_video, photo)
    
    def update_video(self, photo):
        self.video_label.config(image=photo)
        self.video_label.image = photo
    
    def update_prediction(self, prediction, confidence):
        self.prediction_label.config(text=prediction.upper() if prediction else "---")
        self.confidence_var.set(confidence * 100)
        self.confidence_label.config(text=f"{confidence*100:.1f}%")
    
    def update_sentence(self):
        """Update the sentence display"""
        sentence_str = ' '.join(self.detector.sentence)
        self.sentence_text.delete(1.0, tk.END)
        self.sentence_text.insert(1.0, sentence_str)
    
    def on_closing(self):
        self.running = False
        if self.detector.cap:
            self.detector.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()