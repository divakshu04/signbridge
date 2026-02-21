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
        self.sequence = deque(maxlen=self.sequence_length)
        
        # Real-time prediction smoothing
        self.prediction_buffer = deque(maxlen=10)
        self.current_prediction = ""
        self.confidence = 0.0
        
        # Confidence threshold
        self.threshold = 0.35
        
        # Sentence building
        self.sentence = []
        self.last_prediction = ""
        self.same_prediction_count = 0
        self.frames_to_confirm = 12
        
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
                    self.mp_drawing.DrawingSpec(color=(121, 252, 202), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                )
    
    def predict_sign_realtime(self):
        """Real-time prediction with sliding window"""
        if len(self.sequence) == self.sequence_length:
            input_sequence = np.array(list(self.sequence))
            recent_frames = list(self.sequence)[-10:]
            hand_detected_count = sum(1 for frame in recent_frames if np.any(frame))
            
            if hand_detected_count >= 6:
                res = self.model.predict(np.expand_dims(input_sequence, axis=0), verbose=0)[0]
                predicted_idx = np.argmax(res)
                confidence = res[predicted_idx]
                
                if confidence > self.threshold:
                    predicted_sign = self.actions[predicted_idx]
                    self.prediction_buffer.append(predicted_sign)
                    
                    if len(self.prediction_buffer) >= 4:
                        prediction_counts = Counter(self.prediction_buffer)
                        most_common_sign, count = prediction_counts.most_common(1)[0]
                        
                        if count >= 2:
                            self.current_prediction = most_common_sign
                            self.confidence = confidence
                            
                            if most_common_sign == self.last_prediction:
                                self.same_prediction_count += 1
                            else:
                                self.same_prediction_count = 0
                                self.last_prediction = most_common_sign
                            
                            if self.same_prediction_count == self.frames_to_confirm:
                                if len(self.sentence) == 0 or self.sentence[-1] != most_common_sign:
                                    self.sentence.append(most_common_sign)
                                    self.same_prediction_count = 0
            else:
                self.current_prediction = ""
                self.confidence = 0.0
                self.prediction_buffer.clear()
        
        return self.current_prediction, self.confidence


class SignLanguageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SignBridge - Sign Language to Text")
        self.root.geometry("1400x800")
        
        # Modern color scheme
        self.colors = {
            'bg_primary': '#0A0E27',      # Deep navy blue
            'bg_secondary': '#1A1F3A',    # Lighter navy
            'bg_card': '#16213E',         # Card background
            'accent_primary': '#4ECCA3',  # Teal green
            'accent_secondary': '#3D84A8', # Blue
            'text_primary': '#EEEEEE',    # Light gray
            'text_secondary': '#A4B0C0',  # Muted gray
            'success': '#4ECCA3',         # Green
            'warning': '#FFB800',         # Yellow
            'danger': '#FF6B6B',          # Red
            'gradient_start': '#667eea',  # Purple
            'gradient_end': '#764ba2'     # Deep purple
        }
        
        self.root.configure(bg=self.colors['bg_primary'])
        
        # Initialize detector
        self.detector = SignLanguageDetector()
        
        # Setup GUI
        self.setup_gui()
        
        # Threading
        self.running = False
        
    def setup_gui(self):
        # ===== HEADER =====
        header_frame = tk.Frame(self.root, bg=self.colors['bg_secondary'], height=80)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_frame.pack_propagate(False)
        
        # Logo/Icon
        logo_frame = tk.Frame(header_frame, bg=self.colors['bg_secondary'])
        logo_frame.pack(side=tk.LEFT, padx=30, pady=15)
        
        logo_label = tk.Label(
            logo_frame,
            text="🤟",
            font=("Segoe UI Emoji", 32),
            bg=self.colors['bg_secondary'],
            fg=self.colors['accent_primary']
        )
        logo_label.pack()
        
        # Title
        title_frame = tk.Frame(header_frame, bg=self.colors['bg_secondary'])
        title_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=15)
        
        title_label = tk.Label(
            title_frame,
            text="SignBridge",
            font=("Segoe UI", 28, "bold"),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        )
        title_label.pack(anchor='w')
        
        subtitle_label = tk.Label(
            title_frame,
            text="Real-time Sign Language Detection • AI-Powered Translation",
            font=("Segoe UI", 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary']
        )
        subtitle_label.pack(anchor='w')
        
        # Status indicator
        status_frame = tk.Frame(header_frame, bg=self.colors['bg_secondary'])
        status_frame.pack(side=tk.RIGHT, padx=30, pady=20)
        
        self.header_status = tk.Label(
            status_frame,
            text="● Offline",
            font=("Segoe UI", 11),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary']
        )
        self.header_status.pack()
        
        # ===== MAIN CONTENT =====
        main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=25, pady=(0, 20))
        
        # ===== LEFT PANEL - VIDEO =====
        left_panel = tk.Frame(main_container, bg=self.colors['bg_primary'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        # Video card
        video_card = tk.Frame(left_panel, bg=self.colors['bg_card'], relief=tk.FLAT)
        video_card.pack(fill=tk.BOTH, expand=True)
        
        # Video card header
        video_header = tk.Frame(video_card, bg=self.colors['bg_card'], height=50)
        video_header.pack(fill=tk.X, padx=20, pady=(15, 5))
        video_header.pack_propagate(False)
        
        tk.Label(
            video_header,
            text="📹 Live Camera Feed",
            font=("Segoe UI", 13, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side=tk.LEFT)
        
        # Video display
        video_container = tk.Frame(video_card, bg='#000000', relief=tk.FLAT)
        video_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        self.video_label = tk.Label(
            video_container,
            bg='#000000',
            text="Camera Not Active\n\nClick 'Start Detection' to begin",
            font=("Segoe UI", 14),
            fg=self.colors['text_secondary']
        )
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # ===== RIGHT PANEL =====
        right_panel = tk.Frame(main_container, bg=self.colors['bg_primary'], width=450)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(15, 0))
        right_panel.pack_propagate(False)
        
        # ===== CURRENT PREDICTION CARD =====
        pred_card = tk.Frame(right_panel, bg=self.colors['bg_card'], relief=tk.FLAT)
        pred_card.pack(fill=tk.X, pady=(0, 15))
        
        # Prediction header
        pred_header = tk.Frame(pred_card, bg=self.colors['bg_card'])
        pred_header.pack(fill=tk.X, padx=20, pady=(15, 5))
        
        tk.Label(
            pred_header,
            text="🎯 Current Sign",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side=tk.LEFT)
        
        # Prediction display
        pred_display = tk.Frame(pred_card, bg=self.colors['bg_card'])
        pred_display.pack(fill=tk.X, padx=20, pady=15)
        
        self.prediction_label = tk.Label(
            pred_display,
            text="---",
            font=("Segoe UI", 48, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['accent_primary'],
            height=2
        )
        self.prediction_label.pack()
        
        # Confidence section
        conf_frame = tk.Frame(pred_card, bg=self.colors['bg_card'])
        conf_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        conf_label_frame = tk.Frame(conf_frame, bg=self.colors['bg_card'])
        conf_label_frame.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(
            conf_label_frame,
            text="Confidence Level",
            font=("Segoe UI", 10),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary']
        ).pack(side=tk.LEFT)
        
        self.confidence_label = tk.Label(
            conf_label_frame,
            text="0%",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['accent_primary']
        )
        self.confidence_label.pack(side=tk.RIGHT)
        
        # Custom styled progress bar
        progress_container = tk.Frame(conf_frame, bg=self.colors['bg_secondary'], height=8)
        progress_container.pack(fill=tk.X)
        progress_container.pack_propagate(False)
        
        self.confidence_var = tk.DoubleVar()
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor=self.colors['bg_secondary'],
            background=self.colors['accent_primary'],
            bordercolor=self.colors['bg_secondary'],
            lightcolor=self.colors['accent_primary'],
            darkcolor=self.colors['accent_primary'],
            thickness=8
        )
        
        self.confidence_bar = ttk.Progressbar(
            progress_container,
            variable=self.confidence_var,
            maximum=100,
            mode='determinate',
            style="Custom.Horizontal.TProgressbar"
        )
        self.confidence_bar.pack(fill=tk.BOTH, expand=True)
        
        # ===== DETECTED SEQUENCE CARD =====
        sequence_card = tk.Frame(right_panel, bg=self.colors['bg_card'], relief=tk.FLAT)
        sequence_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Sequence header
        sequence_header = tk.Frame(sequence_card, bg=self.colors['bg_card'])
        sequence_header.pack(fill=tk.X, padx=20, pady=(15, 5))
        
        tk.Label(
            sequence_header,
            text="📝 Detected Words",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side=tk.LEFT)
        
        clear_btn = tk.Button(
            sequence_header,
            text="✕ Clear",
            command=self.clear_sentence,
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['danger'],
            fg='#FFFFFF',
            relief=tk.FLAT,
            cursor="hand2",
            padx=15,
            pady=5,
            borderwidth=0,
            activebackground='#ff5252',
            activeforeground='#FFFFFF'
        )
        clear_btn.pack(side=tk.RIGHT)
        
        # Sentence display
        sentence_container = tk.Frame(sequence_card, bg=self.colors['bg_card'])
        sentence_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        scrollbar = tk.Scrollbar(sentence_container, bg=self.colors['bg_secondary'])
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.sentence_text = tk.Text(
            sentence_container,
            font=("Segoe UI", 14),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            relief=tk.FLAT,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            padx=15,
            pady=15,
            spacing1=5,
            spacing3=5,
            borderwidth=0,
            insertbackground=self.colors['accent_primary']
        )
        self.sentence_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.sentence_text.yview)
        
        # ===== INSTRUCTIONS CARD =====
        instructions_card = tk.Frame(right_panel, bg=self.colors['bg_card'], relief=tk.FLAT)
        instructions_card.pack(fill=tk.X)
        
        inst_header = tk.Frame(instructions_card, bg=self.colors['bg_card'])
        inst_header.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        tk.Label(
            inst_header,
            text="💡 Quick Guide",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side=tk.LEFT)
        
        instructions = tk.Text(
            instructions_card,
            height=6,
            font=("Segoe UI", 9),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary'],
            relief=tk.FLAT,
            wrap=tk.WORD,
            padx=20,
            borderwidth=0
        )
        instructions.pack(fill=tk.X, pady=(0, 15))
        instructions.insert(1.0,
            "• Predictions update continuously in real-time\n"
            "• Hold each sign steady for ~0.5 seconds\n"
            "• Words are added automatically when detected\n"
            "• System works with both static and dynamic signs\n"
            "• For motion-based signs, repeat gesture 2-3 times\n"
            "• Ensure good lighting and clear hand visibility"
        )
        instructions.config(state=tk.DISABLED)
        
        # ===== CONTROL BUTTONS =====
        button_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        button_container.pack(fill=tk.X, padx=25, pady=(0, 25))
        
        button_frame = tk.Frame(button_container, bg=self.colors['bg_primary'])
        button_frame.pack()
        
        # Start button
        self.start_button = tk.Button(
            button_frame,
            text="▶  Start Detection",
            command=self.start_detection,
            font=("Segoe UI", 13, "bold"),
            bg=self.colors['accent_primary'],
            fg='#FFFFFF',
            width=18,
            height=2,
            relief=tk.FLAT,
            cursor="hand2",
            borderwidth=0,
            activebackground='#3da889',
            activeforeground='#FFFFFF'
        )
        self.start_button.pack(side=tk.LEFT, padx=8)
        
        # Stop button
        self.stop_button = tk.Button(
            button_frame,
            text="⏹  Stop Detection",
            command=self.stop_detection,
            font=("Segoe UI", 13, "bold"),
            bg=self.colors['danger'],
            fg='#FFFFFF',
            width=18,
            height=2,
            relief=tk.FLAT,
            cursor="hand2",
            state=tk.DISABLED,
            borderwidth=0,
            activebackground='#ff5252',
            activeforeground='#FFFFFF'
        )
        self.stop_button.pack(side=tk.LEFT, padx=8)
        
        # Add hover effects
        self.add_button_hover(self.start_button, self.colors['accent_primary'], '#3da889')
        self.add_button_hover(self.stop_button, self.colors['danger'], '#ff5252')
        self.add_button_hover(clear_btn, self.colors['danger'], '#ff5252')
        
    def add_button_hover(self, button, normal_color, hover_color):
        """Add hover effect to buttons"""
        def on_enter(e):
            if button['state'] != 'disabled':
                button['background'] = hover_color
        
        def on_leave(e):
            if button['state'] != 'disabled':
                button['background'] = normal_color
        
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        
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
            self.start_button.config(state=tk.DISABLED, bg=self.colors['text_secondary'])
            self.stop_button.config(state=tk.NORMAL, bg=self.colors['danger'])
            self.header_status.config(text="● Live", fg=self.colors['success'])
            self.video_label.config(text="")
            
            detection_thread = threading.Thread(target=self.detection_loop)
            detection_thread.daemon = True
            detection_thread.start()
    
    def stop_detection(self):
        self.running = False
        if self.detector.cap:
            self.detector.cap.release()
        self.start_button.config(state=tk.NORMAL, bg=self.colors['accent_primary'])
        self.stop_button.config(state=tk.DISABLED, bg=self.colors['text_secondary'])
        self.header_status.config(text="● Offline", fg=self.colors['text_secondary'])
        self.video_label.config(
            image='',
            text="Camera Not Active\n\nClick 'Start Detection' to begin",
            bg='#000000'
        )
    
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
                
                frame = cv2.flip(frame, 1)
                image, results = self.detector.mediapipe_detection(frame, hands)
                self.detector.draw_styled_landmarks(image, results)
                
                keypoints = self.detector.extract_keypoints(results)
                hand_present = np.any(keypoints)
                self.detector.sequence.append(keypoints)
                
                prediction, confidence = self.detector.predict_sign_realtime()
                
                self.root.after(0, self.update_prediction, prediction, confidence)
                self.root.after(0, self.update_sentence)
                
                # Modern overlay
                hand_text = "HAND DETECTED" if hand_present else "NO HAND"
                hand_color = (78, 204, 163) if hand_present else (255, 107, 107)
                
                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (300, 90), (10, 14, 39), -1)
                cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                
                cv2.putText(image, hand_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2, cv2.LINE_AA)
                
                if self.detector.current_prediction and hand_present:
                    cv2.putText(image, f'{self.detector.current_prediction.upper()}', (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (78, 204, 163), 3, cv2.LINE_AA)
                    
                    progress = min(100, (self.detector.same_prediction_count / self.detector.frames_to_confirm) * 100)
                    bar_width, bar_height, bar_x, bar_y = 250, 12, 15, 100
                    
                    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (26, 31, 62), -1)
                    cv2.rectangle(image, (bar_x, bar_y), (bar_x + int(bar_width * progress / 100), bar_y + bar_height), (78, 204, 163), -1)
                    cv2.putText(image, 'WORD LOCK', (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (164, 176, 192), 1, cv2.LINE_AA)
                else:
                    if not hand_present:
                        cv2.putText(image, 'Show hand to start...', (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (164, 176, 192), 2, cv2.LINE_AA)
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = image.resize((700, 525))
                photo = ImageTk.PhotoImage(image)
                
                self.root.after(0, self.update_video, photo)
    
    def update_video(self, photo):
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo
    
    def update_prediction(self, prediction, confidence):
        if prediction:
            self.prediction_label.config(text=prediction.upper(), fg=self.colors['accent_primary'])
        else:
            self.prediction_label.config(text="---", fg=self.colors['text_secondary'])
        
        self.confidence_var.set(confidence * 100)
        self.confidence_label.config(text=f"{confidence*100:.1f}%")
        
        if confidence > 0.7:
            self.confidence_label.config(fg=self.colors['success'])
        elif confidence > 0.4:
            self.confidence_label.config(fg=self.colors['warning'])
        else:
            self.confidence_label.config(fg=self.colors['text_secondary'])
    
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