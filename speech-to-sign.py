import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import os
import cv2
import time

import speech_recognition as sr


# =========================
# SPEECH TO SIGN CONVERTER
# =========================
class SpeechToSignConverter:
    def __init__(self):
        print("Initializing Speech Recognition...")
        self.recognizer = sr.Recognizer()
        
        try:
            self.microphone = sr.Microphone()
            print(f"Using default microphone")
        except Exception as e:
            print(f"Error initializing microphone: {e}")
            print("\nAvailable microphones:")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"  {index}: {name}")
            self.microphone = sr.Microphone()

            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Supported words (same 29 words)
        self.supported_words = [
            'hello', 'thanks', 'sorry', 'please', 'yes', 'no',
            'i', 'you', 'name', 'time', 'what', 'where', 'how',
            'help', 'learn', 'work', 'eat', 'drink', 'home',
            'good', 'bad', 'happy', 'sad', 'tired',
            '1', '2', '3', '4', '5'
        ]
        
        self.signs_folder = "sign_videos"
        self.recognized_sentence = []
        
        self.recognizer.energy_threshold = 2000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.pause_threshold = 0.9
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.6
        
        print("Speech Recognition ready!")
        
        if not os.path.exists(self.signs_folder):
            os.makedirs(self.signs_folder)
            print(f"Created folder: {self.signs_folder}")

    def get_sign_media_path(self, word):
        """Get path to sign video or image"""
        word_lower = word.lower()
        
        # Check for video formats
        for ext in [".mp4", ".avi", ".mov"]:
            path = os.path.join(self.signs_folder, word_lower + ext)
            if os.path.exists(path):
                return path, "video"
        
        # Check for image formats
        for ext in [".jpg", ".jpeg", ".png"]:
            path = os.path.join(self.signs_folder, word_lower + ext)
            if os.path.exists(path):
                return path, "image"
        
        return None, None


# =========================
# MODERN SPEECH TO SIGN GUI
# =========================
class SpeechToSignGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SignBridge - Speech to Sign Language")
        self.root.geometry("1400x800")
        
        # Modern color scheme (matching app_modern.py)
        self.colors = {
            'bg_primary': '#0A0E27',
            'bg_secondary': '#1A1F3A',
            'bg_card': '#16213E',
            'accent_primary': '#4ECCA3',
            'accent_secondary': '#3D84A8',
            'text_primary': '#EEEEEE',
            'text_secondary': '#A4B0C0',
            'success': '#4ECCA3',
            'warning': '#FFB800',
            'danger': '#FF6B6B',
            'gradient_start': '#667eea',
            'gradient_end': '#764ba2'
        }
        
        self.root.configure(bg=self.colors['bg_primary'])

        self.converter = SpeechToSignConverter()
        self.listening = False
        
        # Video playback control
        self.current_video_cap = None
        self.video_lock = threading.Lock()
        self.stop_background_listening = threading.Event()

        self.build_ui()

    def build_ui(self):
        # ===== HEADER =====
        header_frame = tk.Frame(self.root, bg=self.colors['bg_secondary'], height=80)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_frame.pack_propagate(False)
        
        # Logo/Icon
        logo_frame = tk.Frame(header_frame, bg=self.colors['bg_secondary'])
        logo_frame.pack(side=tk.LEFT, padx=30, pady=15)
        
        logo_label = tk.Label(
            logo_frame,
            text="🎤",
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
            text="Speech to Sign Language • Real-time Voice Recognition",
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
        
        # ===== LEFT PANEL - SIGN VIDEO DISPLAY =====
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
            text="🎬 Sign Language Output",
            font=("Segoe UI", 13, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side=tk.LEFT)
        
        # Sign display container
        sign_container = tk.Frame(video_card, bg='#000000', relief=tk.FLAT)
        sign_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 15))
        
        self.sign_label = tk.Label(
            sign_container,
            text="No Video Playing\n\n🎤 Start listening to see sign language",
            font=("Segoe UI", 14),
            bg='#000000',
            fg=self.colors['text_secondary']
        )
        self.sign_label.pack(expand=True, fill=tk.BOTH)
        
        # Word label (below video)
        word_container = tk.Frame(video_card, bg=self.colors['bg_card'])
        word_container.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(
            word_container,
            text="Current Word:",
            font=("Segoe UI", 10),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary']
        ).pack()
        
        self.word_label = tk.Label(
            word_container,
            text="---",
            font=("Segoe UI", 42, "bold"),
            fg=self.colors['accent_primary'],
            bg=self.colors['bg_card'],
            height=1
        )
        self.word_label.pack(pady=5)
        
        # ===== RIGHT PANEL =====
        right_panel = tk.Frame(main_container, bg=self.colors['bg_primary'], width=450)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(15, 0))
        right_panel.pack_propagate(False)
        
        # ===== MICROPHONE STATUS CARD =====
        mic_card = tk.Frame(right_panel, bg=self.colors['bg_card'], relief=tk.FLAT)
        mic_card.pack(fill=tk.X, pady=(0, 15))
        
        mic_header = tk.Frame(mic_card, bg=self.colors['bg_card'])
        mic_header.pack(fill=tk.X, padx=20, pady=(15, 5))
        
        tk.Label(
            mic_header,
            text="🎙️ Microphone Status",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side=tk.LEFT)
        
        mic_display = tk.Frame(mic_card, bg=self.colors['bg_card'])
        mic_display.pack(fill=tk.X, padx=20, pady=15)
        
        self.mic_indicator = tk.Label(
            mic_display,
            text="● OFFLINE",
            font=("Segoe UI", 20, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary']
        )
        self.mic_indicator.pack()
        
        self.status_label = tk.Label(
            mic_display,
            text="Ready to listen",
            font=("Segoe UI", 10),
            bg=self.colors['bg_card'],
            fg=self.colors['text_secondary']
        )
        self.status_label.pack(pady=(5, 0))
        
        # ===== LIVE TRANSCRIPTION CARD =====
        transcribe_card = tk.Frame(right_panel, bg=self.colors['bg_card'], relief=tk.FLAT)
        transcribe_card.pack(fill=tk.X, pady=(0, 15))
        
        transcribe_header = tk.Frame(transcribe_card, bg=self.colors['bg_card'])
        transcribe_header.pack(fill=tk.X, padx=20, pady=(15, 5))
        
        tk.Label(
            transcribe_header,
            text="📝 Speech Recognized",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side=tk.LEFT)
        
        transcribe_container = tk.Frame(transcribe_card, bg=self.colors['bg_card'])
        transcribe_container.pack(fill=tk.BOTH, padx=20, pady=(10, 15))
        
        self.transcription_text = tk.Text(
            transcribe_container,
            height=4,
            font=("Segoe UI", 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            relief=tk.FLAT,
            wrap=tk.WORD,
            padx=12,
            pady=10,
            borderwidth=0,
            insertbackground=self.colors['accent_primary']
        )
        self.transcription_text.pack(fill=tk.BOTH, expand=True)
        
        # ===== WORD SEQUENCE CARD =====
        sequence_card = tk.Frame(right_panel, bg=self.colors['bg_card'], relief=tk.FLAT)
        sequence_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        sequence_header = tk.Frame(sequence_card, bg=self.colors['bg_card'])
        sequence_header.pack(fill=tk.X, padx=20, pady=(15, 5))
        
        tk.Label(
            sequence_header,
            text="✅ Word Sequence",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side=tk.LEFT)
        
        clear_btn = tk.Button(
            sequence_header,
            text="✕ Clear",
            command=self.clear_sequence,
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
        
        sequence_container = tk.Frame(sequence_card, bg=self.colors['bg_card'])
        sequence_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        scrollbar = tk.Scrollbar(sequence_container, bg=self.colors['bg_secondary'])
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.sequence_text = tk.Text(
            sequence_container,
            font=("Segoe UI", 13),
            bg=self.colors['bg_secondary'],
            fg=self.colors['accent_primary'],
            relief=tk.FLAT,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            padx=15,
            pady=12,
            spacing1=5,
            spacing3=5,
            borderwidth=0,
            insertbackground=self.colors['accent_primary']
        )
        self.sequence_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.sequence_text.yview)
        
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
            height=7,
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
            "• Click 'Start Listening' to activate voice recognition\n"
            "• Speak clearly and naturally in English\n"
            "• Videos play instantly when words are detected\n"
            "• System recognizes 29 common sign language words\n"
            "• Brief pauses between words improve accuracy\n"
            "• Ensure quiet environment and good microphone\n"
            "• Internet connection required for speech recognition"
        )
        instructions.config(state=tk.DISABLED)
        
        # ===== CONTROL BUTTONS =====
        button_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        button_container.pack(fill=tk.X, padx=25, pady=(0, 25))
        
        button_frame = tk.Frame(button_container, bg=self.colors['bg_primary'])
        button_frame.pack()
        
        # Start button
        self.listen_button = tk.Button(
            button_frame,
            text="🎤  Start Listening",
            command=self.start_listening,
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
        self.listen_button.pack(side=tk.LEFT, padx=8)
        
        # Stop button
        self.stop_button = tk.Button(
            button_frame,
            text="⏹  Stop Listening",
            command=self.stop_listening,
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
        self.add_button_hover(self.listen_button, self.colors['accent_primary'], '#3da889')
        self.add_button_hover(self.stop_button, self.colors['danger'], '#ff5252')
        self.add_button_hover(clear_btn, self.colors['danger'], '#ff5252')
        
        # ===== SUPPORTED WORDS FOOTER =====
        words_card = tk.Frame(self.root, bg=self.colors['bg_card'], relief=tk.FLAT)
        words_card.pack(fill=tk.X, padx=25, pady=(0, 25))
        
        words_header = tk.Frame(words_card, bg=self.colors['bg_card'])
        words_header.pack(fill=tk.X, padx=20, pady=(12, 8))
        
        tk.Label(
            words_header,
            text="Supported Words (29)",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary']
        ).pack(side=tk.LEFT)
        
        words_text = ', '.join(self.converter.supported_words)
        words_label = tk.Label(
            words_card,
            text=words_text,
            font=("Segoe UI", 9),
            bg=self.colors['bg_card'],
            fg=self.colors['accent_secondary'],
            wraplength=1300,
            justify=tk.LEFT
        )
        words_label.pack(fill=tk.X, padx=20, pady=(0, 12))

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

    def clear_sequence(self):
        """Clear word sequence"""
        self.converter.recognized_sentence = []
        self.sequence_text.delete(1.0, tk.END)

    def start_listening(self):
        """Start continuous listening"""
        if not self.listening:
            self.listening = True
            self.listen_button.config(state=tk.DISABLED, bg=self.colors['text_secondary'])
            self.stop_button.config(state=tk.NORMAL, bg=self.colors['danger'])
            self.mic_indicator.config(text="● LISTENING", fg=self.colors['success'])
            self.header_status.config(text="● Live", fg=self.colors['success'])
            self.update_status("Calibrating microphone...")
            
            # Clear the "No Video" message
            self.sign_label.config(text="")
            
            # Start background listening
            threading.Thread(target=self.start_background_listening, daemon=True).start()

    def stop_listening(self):
        """Stop listening"""
        self.listening = False
        self.stop_background_listening.set()
        self.listen_button.config(state=tk.NORMAL, bg=self.colors['accent_primary'])
        self.stop_button.config(state=tk.DISABLED, bg=self.colors['text_secondary'])
        self.mic_indicator.config(text="● OFFLINE", fg=self.colors['text_secondary'])
        self.header_status.config(text="● Offline", fg=self.colors['text_secondary'])
        self.update_status("Stopped")
        
        # Show "No Video" message
        self.sign_label.config(
            image='',
            text="No Video Playing\n\n🎤 Start listening to see sign language",
            bg='#000000'
        )

    def start_background_listening(self):
        """Start background listening"""
        self.stop_background_listening.clear()
        
        # Calibrate microphone
        with self.converter.microphone as source:
            print("Calibrating...")
            self.converter.recognizer.adjust_for_ambient_noise(source, duration=3)
            print("Ready!")
            self.root.after(0, self.update_status, "Listening for speech...")
        
        # Start background listening
        stop_listening = self.converter.recognizer.listen_in_background(
            self.converter.microphone,
            self.audio_callback,
            phrase_time_limit=4
        )
        
        self.stop_listening_func = stop_listening
        self.stop_background_listening.wait()
        stop_listening(wait_for_stop=False)

    def audio_callback(self, recognizer, audio):
        """INSTANT audio processing"""
        if not self.listening:
            return
        
        audio = sr.AudioData(
            audio.get_raw_data(convert_rate=16000, convert_width=2),
            16000,
            2
        )
        
        try:
            # Quick recognition
            text = recognizer.recognize_google(audio).lower()
            print(f"⚡ Heard: '{text}'")
            
            # Update UI
            self.root.after(0, self.update_transcription, text)
            self.root.after(0, self.update_status, "Processing...")
            
            # Extract supported words
            words = text.split()
            found_words = [w for w in words if w in self.converter.supported_words]
            
            if found_words:
                print(f"✓ Found: {found_words}")
                
                # Play FIRST word instantly
                first_word = found_words[0]
                
                threading.Thread(
                    target=self.display_sign_instant,
                    args=(first_word,),
                    daemon=True
                ).start()
                
                # Add all words to sequence
                for word in found_words:
                    self.converter.recognized_sentence.append(word)
                self.root.after(0, self.update_sequence)
                
                self.root.after(0, self.update_status, "Listening for speech...")
            else:
                self.root.after(0, self.update_status, "Listening for speech...")
                
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            self.root.after(0, self.update_status, "⚠️ Internet error")
        except Exception as e:
            print(f"Error: {e}")

    def display_sign_instant(self, word):
        """INSTANT sign display"""
        # Stop current video
        with self.video_lock:
            if self.current_video_cap:
                self.current_video_cap.release()
                self.current_video_cap = None
        
        # Update word label
        self.root.after(0, lambda: self.word_label.config(text=word.upper()))
        
        # Get media path
        media_path, media_type = self.converter.get_sign_media_path(word)
        
        if not media_path:
            self.root.after(0, self.show_no_media, word)
            return
        
        if media_type == "image":
            self.root.after(0, self.display_image, media_path)
        elif media_type == "video":
            self.play_video_instant(media_path)

    def show_no_media(self, word):
        """Show no media message"""
        self.sign_label.config(
            text=f"⚠️ No media found for '{word}'\n\nPlease add {word}.mp4 or {word}.jpg\nto the 'sign_videos' folder",
            image='',
            fg=self.colors['warning'],
            bg='#000000'
        )

    def display_image(self, image_path):
        """Display sign image"""
        try:
            img = Image.open(image_path)
            img = img.resize((600, 500), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.sign_label.config(image=photo, text='', bg='#000000')
            self.sign_label.image = photo
        except Exception as e:
            print(f"Image error: {e}")

    def play_video_instant(self, video_path):
        """Play video with instant interrupt capability"""
        with self.video_lock:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Can't open: {video_path}")
                return
            
            self.current_video_cap = cap
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps > 60:
                fps = 30
            frame_delay = 1.0 / fps
        
        # Play video
        while True:
            with self.video_lock:
                if self.current_video_cap != cap:
                    cap.release()
                    print("Video interrupted")
                    return
                
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    self.current_video_cap = None
                    return
            
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (600, 500))
                img = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(img)
                
                self.sign_label.config(image=photo, text='', bg='#000000')
                self.sign_label.image = photo
                
                time.sleep(frame_delay)
                
            except Exception as e:
                print(f"Frame error: {e}")
                break

    def update_status(self, text):
        """Update status label"""
        self.status_label.config(text=text)

    def update_transcription(self, text):
        """Update transcription"""
        self.transcription_text.insert(tk.END, text + "\n")
        self.transcription_text.see(tk.END)
        
        # Keep last 6 lines
        lines = self.transcription_text.get(1.0, tk.END).split('\n')
        if len(lines) > 8:
            self.transcription_text.delete(1.0, "2.0")

    def update_sequence(self):
        """Update word sequence"""
        sequence_str = ' → '.join(self.converter.recognized_sentence)
        self.sequence_text.delete(1.0, tk.END)
        self.sequence_text.insert(1.0, sequence_str)
        self.sequence_text.see(tk.END)

    def on_closing(self):
        """Clean shutdown"""
        self.listening = False
        self.stop_background_listening.set()
        with self.video_lock:
            if self.current_video_cap:
                self.current_video_cap.release()
        self.root.destroy()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechToSignGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()