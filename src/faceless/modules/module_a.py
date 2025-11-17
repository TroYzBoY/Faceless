"""
🚀 ENHANCED FACE RECOGNITION - OpenCV Only Version
✅ dlib шаардлагагүй
✅ Бүх боломжтой
✅ Windows-д шууд ажиллана
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import pickle
import time
from datetime import datetime
import numpy as np
import os
from collections import Counter
import threading
import json
import platform
import sys
import sqlite3

from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet


class EnhancedFaceRecognitionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 AI Нүүр таних систем - Enhanced")
        self.root.geometry("1200x700")

        # Detect OS
        self.is_macos = platform.system() == 'Darwin'
        self.is_windows = platform.system() == 'Windows'

        # Set macOS-specific configurations
        if self.is_macos:
            # macOS-specific settings
            try:
                # Try to use native macOS appearance
                # Note: Scaling is handled automatically by macOS
                pass
            except:
                pass

        # Color scheme (works on both Windows and macOS)
        self.bg_dark = '#0a0e27'
        self.bg_panel = '#1a1f3a'
        self.fg_primary = '#00ff9f'
        self.fg_secondary = '#ffffff'
        self.fg_muted = '#666699'

        self.root.configure(bg=self.bg_dark)

        # Face data storage
        self.known_face_features = []
        self.known_face_names = []
        self.face_quality_scores = []
        self.data_file = "enhanced_face_data.pkl"
        self.db_path = os.path.join("data", "face_embeddings.db")
        self.db_lock = threading.Lock()
        self.embedding_size = 512
        self.threshold = 0.70  # Adjusted for FaceNet embeddings

        # Initialize modern face detection & embedding models
        self.detector = MTCNN()
        self.facenet = FaceNet()

        # Video capture
        self.video_capture = None
        self.is_capturing = False
        self.current_mode = None
        self.fps = 0

        self.init_database()
        self.setup_ui()
        self.load_data_silent()

    def get_font(self, size, weight='normal'):
        """Get platform-appropriate font"""
        if self.is_macos:
            # macOS fonts - try SF Pro first, fallback to system fonts
            try:
                if weight == 'bold':
                    # Try SF Pro Display, fallback to Helvetica
                    return ('SF Pro Display', size, 'bold')
                else:
                    # Try SF Pro Text, fallback to Helvetica
                    return ('SF Pro Text', size)
            except:
                # Fallback to system fonts
                if weight == 'bold':
                    return ('Helvetica Neue', size, 'bold')
                else:
                    return ('Helvetica Neue', size)
        elif self.is_windows:
            # Windows fonts
            if weight == 'bold':
                return ('Segoe UI', size, 'bold')
            else:
                return ('Segoe UI', size)
        else:
            # Linux/Other
            return ('DejaVu Sans', size, weight)

    def setup_ui(self):
        """Setup modern UI with macOS compatibility"""
        # Title bar
        title_frame = tk.Frame(self.root, bg=self.bg_panel, height=90)
        title_frame.pack(fill='x', pady=(0, 10))

        title_label = tk.Label(
            title_frame,
            text="🚀 TEAM TAM НҮҮР ТАНИХ СИСТЕМ",
            font=self.get_font(26, 'bold'),
            bg=self.bg_panel,
            fg=self.fg_primary
        )
        title_label.pack(pady=15)

        mode_label = tk.Label(
            title_frame,
            text="🟢 ENHANCED MODE (OpenCV + Deep Features)",
            font=self.get_font(10, 'bold'),
            bg=self.bg_panel,
            fg=self.fg_primary
        )
        mode_label.pack()

        # Main container
        main_container = tk.Frame(self.root, bg=self.bg_dark)
        main_container.pack(fill='both', expand=True, padx=20, pady=10)

        # Left panel
        left_panel = tk.Frame(main_container, bg=self.bg_panel, width=380)
        left_panel.pack(side='left', fill='both', padx=(0, 10))

        # Control buttons
        control_frame = tk.LabelFrame(
            left_panel,
            text="⚡ Үндсэн үйлдлүүд",
            font=self.get_font(12, 'bold'),
            bg=self.bg_panel,
            fg=self.fg_secondary,
            padx=15,
            pady=15,
            relief='flat' if self.is_macos else 'groove',
            borderwidth=1
        )
        control_frame.pack(fill='x', pady=10, padx=10)

        self.register_btn = self.create_button(
            control_frame, "🤖 Нүүр бүртгэх", self.start_registration, '#00ff9f')
        self.register_btn.pack(fill='x', pady=5)

        self.recognize_btn = self.create_button(
            control_frame, "🎥 Танилт эхлүүлэх", self.start_recognition, '#00aaff')
        self.recognize_btn.pack(fill='x', pady=5)

        self.stop_btn = self.create_button(
            control_frame, "⏹️ Зогсоох", self.stop_capture, '#ff4466')
        self.stop_btn.pack(fill='x', pady=5)
        self.stop_btn.config(state='disabled')

        # Advanced settings
        advanced_frame = tk.LabelFrame(
            left_panel,
            text="🎛️ Нарийвчилсан тохиргоо",
            font=self.get_font(12, 'bold'),
            bg=self.bg_panel,
            fg=self.fg_secondary,
            padx=15,
            pady=15,
            relief='flat' if self.is_macos else 'groove',
            borderwidth=1
        )
        advanced_frame.pack(fill='x', pady=10, padx=10)

        # Checkbutton colors for macOS
        checkbutton_bg = self.bg_panel
        checkbutton_fg = self.fg_secondary
        checkbutton_select = '#2a2f4a' if not self.is_macos else '#3a3f5a'

        self.multi_angle_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            advanced_frame, text="📐 Олон өнцгөөс авах",
            variable=self.multi_angle_var, bg=checkbutton_bg, fg=checkbutton_fg,
            selectcolor=checkbutton_select, font=self.get_font(10),
            activebackground=checkbutton_bg, activeforeground=checkbutton_fg
        ).pack(anchor='w', pady=3)

        self.quality_filter_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            advanced_frame, text="✨ Чанарын шүүлтүүр",
            variable=self.quality_filter_var, bg=checkbutton_bg, fg=checkbutton_fg,
            selectcolor=checkbutton_select, font=self.get_font(10),
            activebackground=checkbutton_bg, activeforeground=checkbutton_fg
        ).pack(anchor='w', pady=3)

        tk.Label(
            advanced_frame, text="🧠 FaceNet embeddings (идэвхтэй)",
            bg=checkbutton_bg, fg=self.fg_secondary, font=self.get_font(10)
        ).pack(anchor='w', pady=3)

        self.show_confidence_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            advanced_frame, text="📊 Confidence bar",
            variable=self.show_confidence_var, bg=checkbutton_bg, fg=checkbutton_fg,
            selectcolor=checkbutton_select, font=self.get_font(10),
            activebackground=checkbutton_bg, activeforeground=checkbutton_fg
        ).pack(anchor='w', pady=3)

        # Data management
        data_frame = tk.LabelFrame(
            left_panel, text="💾 Дата удирдлага",
            font=self.get_font(12, 'bold'),
            bg=self.bg_panel, fg=self.fg_secondary, padx=15, pady=15,
            relief='flat' if self.is_macos else 'groove',
            borderwidth=1
        )
        data_frame.pack(fill='x', pady=10, padx=10)

        buttons = [
            ("📂 Дата ачаалах", self.load_data, '#9966ff'),
            ("💾 Дата хадгалах", self.save_data, '#9966ff'),
            ("📤 Export JSON", self.export_json, '#ff9500'),
            ("📥 Import зураг", self.import_from_folder, '#ff9500'),
            ("👥 Хүмүүсийг харах", self.show_people_list, '#00aaff'),
            ("📈 Статистик", self.show_statistics, '#00aaff'),
            ("🗑️ Хүн устгах", self.delete_person, '#ff4466'),
        ]

        for text, cmd, color in buttons:
            self.create_button(data_frame, text, cmd,
                               color).pack(fill='x', pady=3)

        # Settings
        settings_frame = tk.LabelFrame(
            left_panel, text="⚙️ Тохиргоо",
            font=self.get_font(12, 'bold'),
            bg=self.bg_panel, fg=self.fg_secondary, padx=15, pady=15,
            relief='flat' if self.is_macos else 'groove',
            borderwidth=1
        )
        settings_frame.pack(fill='x', pady=10, padx=10)

        tk.Label(settings_frame, text="Threshold утга:",
                 bg=self.bg_panel, fg=self.fg_secondary, font=self.get_font(10)).pack(anchor='w')

        self.threshold_var = tk.DoubleVar(value=self.threshold)
        threshold_slider = ttk.Scale(
            settings_frame, from_=0.50, to=0.85,
            variable=self.threshold_var, orient='horizontal',
            command=self.update_threshold
        )
        threshold_slider.pack(fill='x', pady=5)

        self.threshold_label = tk.Label(
            settings_frame, text=f"Утга: {self.threshold:.2f}",
            bg=self.bg_panel, fg=self.fg_primary, font=self.get_font(9, 'bold')
        )
        self.threshold_label.pack()

        # Status display
        status_frame = tk.LabelFrame(
            left_panel, text="📊 Систем мэдээлэл",
            font=self.get_font(12, 'bold'),
            bg=self.bg_panel, fg=self.fg_secondary, padx=15, pady=15,
            relief='flat' if self.is_macos else 'groove',
            borderwidth=1
        )
        status_frame.pack(fill='both', expand=True, pady=10, padx=10)

        # Use monospace font that works on macOS
        monospace_font = 'Menlo' if self.is_macos else (
            'Consolas' if self.is_windows else 'Monaco')
        self.status_text = tk.Text(
            status_frame, height=12, bg=self.bg_dark, fg=self.fg_primary,
            font=(monospace_font, 9), wrap='word', state='disabled',
            borderwidth=0, highlightthickness=0,
            insertbackground=self.fg_primary
        )
        self.status_text.pack(fill='both', expand=True)

        # Right panel - Video
        right_panel = tk.Frame(main_container, bg=self.bg_panel)
        right_panel.pack(side='right', fill='both', expand=True)

        video_frame = tk.LabelFrame(
            right_panel, text="📹 Видео харагдац",
            font=self.get_font(12, 'bold'),
            bg=self.bg_panel, fg=self.fg_secondary,
            relief='flat' if self.is_macos else 'groove',
            borderwidth=1
        )
        video_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.video_label = tk.Label(
            video_frame, bg=self.bg_dark,
            text="🎥 Видео зогссон байна\n\n✨ Эхлүүлэхийн тулд дээрх товчийг дарна уу",
            font=self.get_font(14), fg=self.fg_muted
        )
        self.video_label.pack(fill='both', expand=True, padx=10, pady=10)

        # Info bar
        info_frame = tk.Frame(right_panel, bg=self.bg_panel, height=40)
        info_frame.pack(fill='x', padx=10, pady=(0, 10))

        self.info_label = tk.Label(
            info_frame, text="⚡ Бэлэн",
            font=self.get_font(10), bg=self.bg_panel, fg=self.fg_primary
        )
        self.info_label.pack(side='left', padx=10, pady=5)

        self.update_status_display()

    def create_button(self, parent, text, command, color):
        """Create styled button with macOS compatibility"""
        # Adjust button style for macOS
        if self.is_macos:
            relief = 'flat'
            borderwidth = 1
            highlightthickness = 0
        else:
            relief = 'flat'
            borderwidth = 0
            highlightthickness = 0

        btn = tk.Button(
            parent, text=text, command=command,
            bg=color, fg='#ffffff', font=self.get_font(11, 'bold'),
            relief=relief, cursor='hand2', height=2,
            activebackground=self.lighten_color(color),
            activeforeground='#ffffff',
            borderwidth=borderwidth,
            highlightthickness=highlightthickness
        )
        btn.bind('<Enter>', lambda e: btn.config(bg=self.lighten_color(color)))
        btn.bind('<Leave>', lambda e: btn.config(bg=color))
        return btn

    def show_warning_async(self, title, message):
        """Thread-safe warning dialog"""
        self.root.after(0, lambda: messagebox.showwarning(title, message))

    def lighten_color(self, color):
        """Lighten hex color"""
        colors = {
            '#00ff9f': '#33ffb3', '#00aaff': '#33bbff',
            '#ff4466': '#ff6688', '#9966ff': '#aa77ff',
            '#ff9500': '#ffaa33'
        }
        return colors.get(color, color)

    def init_database(self):
        """Initialize SQLite storage for embeddings"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        with self.db_lock:
            self.db_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    quality REAL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self.db_conn.commit()
        self.refresh_memory_from_db()

    def refresh_memory_from_db(self):
        """Load all embeddings from SQLite into memory"""
        with self.db_lock:
            cursor = self.db_conn.execute(
                "SELECT name, embedding, embedding_dim, quality FROM face_embeddings"
            )
            rows = cursor.fetchall()

        features = []
        names = []
        qualities = []

        for name, embedding_blob, embedding_dim, quality in rows:
            if embedding_blob is None:
                continue
            vector = np.frombuffer(embedding_blob, dtype=np.float32)
            if embedding_dim and vector.size > embedding_dim:
                vector = vector[:embedding_dim]
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            features.append(vector)
            names.append(name)
            qualities.append(quality if quality is not None else 50.0)

        self.known_face_features = features
        self.known_face_names = names
        self.face_quality_scores = qualities

    def save_embedding_to_db(self, name, embedding, quality):
        """Persist a single embedding to SQLite"""
        embedding_array = np.asarray(embedding, dtype=np.float32)
        if embedding_array.size == 0:
            return

        record = (
            name,
            embedding_array.tobytes(),
            int(embedding_array.size),
            float(quality),
            datetime.now().isoformat()
        )

        with self.db_lock:
            self.db_conn.execute(
                """
                INSERT INTO face_embeddings (name, embedding, embedding_dim, quality, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                record
            )
            self.db_conn.commit()

        self.refresh_memory_from_db()

    def bulk_insert_embeddings(self, names, features, qualities):
        """Insert multiple embeddings in one transaction"""
        if not names or not features:
            return

        payload = []
        for name, feature, quality in zip(names, features, qualities):
            feature_array = np.asarray(feature, dtype=np.float32)
            if feature_array.size == 0:
                continue
            payload.append((
                name,
                feature_array.tobytes(),
                int(feature_array.size),
                float(quality),
                datetime.now().isoformat()
            ))

        if not payload:
            return

        with self.db_lock:
            self.db_conn.executemany(
                """
                INSERT INTO face_embeddings (name, embedding, embedding_dim, quality, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                payload
            )
            self.db_conn.commit()

        self.refresh_memory_from_db()

    def delete_person_from_db(self, name):
        """Remove all embeddings for a person"""
        with self.db_lock:
            self.db_conn.execute(
                "DELETE FROM face_embeddings WHERE name = ?", (name,)
            )
            self.db_conn.commit()
        self.refresh_memory_from_db()

    def detect_faces_mtcnn(self, frame, min_size=80):
        """Detect faces using MTCNN and return normalized boxes"""
        if frame is None or frame.size == 0:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb)
        faces = []

        for det in detections:
            confidence = det.get('confidence', 0)
            if confidence < 0.90:
                continue

            x, y, w, h = det.get('box', (0, 0, 0, 0))
            if w < min_size or h < min_size:
                continue

            x = max(0, x)
            y = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            w = max(1, x2 - x)
            h = max(1, y2 - y)

            faces.append({
                'box': (x, y, w, h),
                'confidence': confidence,
                'keypoints': det.get('keypoints', {})
            })

        return faces

    def prewhiten(self, x):
        """Prewhiten image for FaceNet"""
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = (x - mean) / std_adj
        return y

    def extract_facenet_embedding(self, face_image):
        """Generate FaceNet embedding for given face ROI"""
        if face_image is None or face_image.size == 0:
            return None

        try:
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (160, 160))
            normalized = resized.astype(np.float32)
            normalized = self.prewhiten(normalized)
            embedding = self.facenet.embeddings([normalized])[0]
            if embedding is None or embedding.size == 0:
                return None
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return None
            return (embedding / norm).astype(np.float32)
        except Exception:
            return None

    def calculate_face_quality(self, face_image):
        """Calculate quality score"""
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(
                face_image.shape) == 3 else face_image

            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()

            brightness = np.mean(gray)
            brightness_score = 100 - abs(brightness - 128)

            contrast = gray.std()

            quality = min(
                100, (sharpness * 3 + brightness_score + contrast) / 5)
            return max(0, quality)
        except:
            return 50.0

    def compare_features(self, feat1, feat2):
        """Compare features"""
        try:
            # Ensure features are numpy arrays and valid
            feat1 = np.array(feat1, dtype=np.float32)
            feat2 = np.array(feat2, dtype=np.float32)

            # Check for invalid values
            if np.any(np.isnan(feat1)) or np.any(np.isnan(feat2)):
                return 0.0
            if np.any(np.isinf(feat1)) or np.any(np.isinf(feat2)):
                return 0.0

            # Normalize features
            norm1 = np.linalg.norm(feat1)
            norm2 = np.linalg.norm(feat2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            feat1_norm = feat1 / norm1
            feat2_norm = feat2 / norm2

            # Cosine similarity
            cos_sim = np.clip(np.dot(feat1_norm, feat2_norm), -1.0, 1.0)

            # Euclidean distance similarity
            euclidean_dist = np.linalg.norm(feat1_norm - feat2_norm)
            euclidean_sim = 1 / (1 + euclidean_dist)

            # Combined similarity (weighted)
            similarity = 0.7 * cos_sim + 0.3 * euclidean_sim

            # Ensure similarity is in valid range
            return np.clip(similarity, 0.0, 1.0)
        except Exception as e:
            return 0.0

    def start_registration(self):
        """Start registration"""
        if self.is_capturing:
            messagebox.showwarning("Анхааруулга", "Өөр үйлдэл явагдаж байна!")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("✨ Нүүр бүртгэх")
        dialog.geometry("450x280")
        dialog.configure(bg=self.bg_panel)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="👤 Хүний нэр оруулна уу:",
                 font=self.get_font(13, 'bold'),
                 bg=self.bg_panel, fg=self.fg_secondary).pack(pady=25)

        name_entry = tk.Entry(dialog, font=self.get_font(12), width=30,
                              bg='#2a2f4a', fg=self.fg_secondary,
                              insertbackground=self.fg_primary, relief='flat', borderwidth=5)
        name_entry.pack(pady=10)
        name_entry.focus()

        tk.Label(dialog, text="📸 Зургийн тоо:",
                 font=self.get_font(10),
                 bg=self.bg_panel, fg=self.fg_secondary).pack(pady=(10, 5))

        sample_var = tk.IntVar(value=15)
        tk.Spinbox(dialog, from_=8, to=25, textvariable=sample_var,
                   font=self.get_font(11), width=10,
                   bg='#2a2f4a', fg=self.fg_secondary).pack()

        def submit():
            name = name_entry.get().strip()
            if name:
                # Prevent duplicate registration for the same person
                existing_names = [n.lower() for n in self.known_face_names]
                if name.lower() in existing_names:
                    messagebox.showwarning(
                        "Анхааруулга",
                        f"'{name}' нэртэй хүн аль хэдийн бүртгэлтэй байна!"
                    )
                    return
                dialog.destroy()
                self.register_name = name
                self.register_samples = sample_var.get()
                threading.Thread(target=self.register_thread,
                                 daemon=True).start()
            else:
                messagebox.showerror("Алдаа", "Нэр оруулна уу!")

        submit_btn = tk.Button(dialog, text="✓ Эхлүүлэх", command=submit,
                               bg=self.fg_primary, fg=self.bg_dark,
                               font=self.get_font(11, 'bold'),
                               cursor='hand2', height=2, relief='flat',
                               activebackground=self.lighten_color(
                                   self.fg_primary),
                               activeforeground=self.bg_dark)
        submit_btn.pack(pady=15)

        name_entry.bind('<Return>', lambda e: submit())

    def is_face_centered(self, face_rect, frame_shape, center_threshold=0.15):
        """Check if face is centered in frame"""
        x, y, w, h = face_rect
        frame_center_x = frame_shape[1] // 2
        frame_center_y = frame_shape[0] // 2
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Calculate distance from center
        dx = abs(face_center_x - frame_center_x) / frame_shape[1]
        dy = abs(face_center_y - frame_center_y) / frame_shape[0]

        # Check if face is within center threshold
        return dx < center_threshold and dy < center_threshold

    def draw_center_guide(self, frame):
        """Draw center guide lines"""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Draw crosshair
        line_length = 30
        thickness = 2
        color = (100, 100, 255)  # Light blue

        # Horizontal line
        cv2.line(frame, (center_x - line_length, center_y),
                 (center_x + line_length, center_y), color, thickness)
        # Vertical line
        cv2.line(frame, (center_x, center_y - line_length),
                 (center_x, center_y + line_length), color, thickness)

        # Draw center circle
        cv2.circle(frame, (center_x, center_y), 50, color, 2)

    def clean_features(self, features_list):
        """Clean and normalize features list"""
        cleaned_features = []
        for features in features_list:
            try:
                features_array = np.array(features, dtype=np.float32)

                # Remove NaN and Inf
                if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                    continue

                # Normalize
                norm = np.linalg.norm(features_array)
                if norm > 0:
                    features_array = features_array / norm
                else:
                    continue

                cleaned_features.append(features_array)
            except:
                continue

        return cleaned_features

    def face_already_registered(self, new_features, threshold=0.90):
        """Check if features belong to an existing person"""
        if not self.known_face_features:
            return False

        try:
            new_feat = np.array(new_features, dtype=np.float32)
            if np.linalg.norm(new_feat) == 0:
                return False
        except:
            return False

        for existing in self.known_face_features:
            try:
                existing_feat = np.array(existing, dtype=np.float32)
                if np.linalg.norm(existing_feat) == 0:
                    continue
                similarity = self.compare_features(new_feat, existing_feat)
                if similarity >= threshold:
                    return True
            except:
                continue

        return False

    def register_thread(self):
        """Registration thread"""
        self.is_capturing = True
        self.current_mode = 'register'
        self.register_btn.config(state='disabled')
        self.recognize_btn.config(state='disabled')
        self.stop_btn.config(state='normal')

        self.update_status(f"\n🚀 {self.register_name} бүртгэж байна...")
        self.update_status("📍 Нүүрээ камерын төвд байрлуулна уу")
        self.info_label.config(text=f"📸 Бүртгэл: {self.register_name}")

        self.video_capture = cv2.VideoCapture(0)

        features_list = []
        quality_list = []
        count = 0
        face_positions = []
        last_capture = time.time()
        stable_frames = 0
        centered_frames = 0

        while count < self.register_samples and self.is_capturing:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            # Draw center guide
            self.draw_center_guide(frame)

            faces = self.detect_faces_mtcnn(frame, min_size=120)
            current_time = time.time()

            # Process only the largest face (closest to camera)
            if len(faces) > 0:
                # Sort by area (largest first)
                faces = sorted(
                    faces, key=lambda f: f['box'][2] * f['box'][3], reverse=True)
                face_info = faces[0]
                x, y, w, h = face_info['box']
                keypoints = face_info.get('keypoints') or {}
                has_eyes = 'left_eye' in keypoints and 'right_eye' in keypoints

                face_center = (x + w//2, y + h//2)
                is_centered = self.is_face_centered((x, y, w, h), frame.shape)
                is_new_angle = self.is_new_angle(
                    face_center, face_positions) if self.multi_angle_var.get() else True

                face_roi = frame[y:y+h, x:x+w]
                quality = self.calculate_face_quality(face_roi)
                quality_ok = quality > 35 if self.quality_filter_var.get() else True

                # All conditions must be met
                ready = has_eyes and is_centered and is_new_angle and quality_ok

                if ready:
                    color = (0, 255, 0)  # Green - ready
                    stable_frames += 1
                    centered_frames += 1
                else:
                    if not is_centered:
                        color = (0, 165, 255)  # Orange - not centered
                        centered_frames = 0
                    elif not quality_ok:
                        color = (255, 0, 255)  # Magenta - low quality
                        centered_frames = 0
                    elif not has_eyes:
                        color = (0, 255, 255)  # Yellow - no eyes
                        centered_frames = 0
                    else:
                        color = (0, 165, 255)  # Orange - same angle
                        centered_frames = 0
                    stable_frames = 0

                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

                # Draw center indicator
                if is_centered:
                    cv2.circle(frame, face_center, 5, (0, 255, 0), -1)

                # Status text
                status_text = []
                if self.quality_filter_var.get():
                    status_text.append(f"Q: {quality:.0f}%")
                if not is_centered:
                    status_text.append("Центрт байрлуулна уу")
                elif not has_eyes:
                    status_text.append("Нүд харагдахгүй")
                elif ready:
                    status_text.append("✓ Бэлэн")

                if status_text:
                    text = " | ".join(status_text)
                    cv2.putText(frame, text, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Capture when face is centered and stable
                if ready and stable_frames >= 5 and centered_frames >= 3 and current_time - last_capture >= 0.6:
                    features = self.extract_facenet_embedding(face_roi)

                    # Validate features before saving
                    if features is not None and len(features) > 0:
                        try:
                            features_array = np.array(
                                features, dtype=np.float32)
                            if not np.any(np.isnan(features_array)) and not np.any(np.isinf(features_array)):
                                # Normalize feature
                                norm = np.linalg.norm(features_array)
                                if norm > 0:
                                    features_array = features_array / norm
                                    features_list.append(features_array)
                                    quality_list.append(quality)
                                    face_positions.append(face_center)
                                    count += 1
                                    last_capture = current_time
                                    stable_frames = 0
                                    centered_frames = 0

                                    # Flash effect
                                    overlay = frame.copy()
                                    cv2.circle(
                                        overlay, (frame.shape[1]//2, frame.shape[0]//2), 100, (0, 255, 0), -1)
                                    frame = cv2.addWeighted(
                                        frame, 0.6, overlay, 0.4, 0)

                                    self.update_status(
                                        f"📸 {count}/{self.register_samples} - Q: {quality:.0f}%")
                        except:
                            # Skip invalid features
                            pass

            self.draw_progress(frame, count, self.register_samples)
            self.display_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()

        if count >= 3:
            # Clean features before saving - match with quality and positions
            cleaned_data = []
            for i, features in enumerate(features_list):
                try:
                    features_array = np.array(features, dtype=np.float32)
                    if not np.any(np.isnan(features_array)) and not np.any(np.isinf(features_array)):
                        norm = np.linalg.norm(features_array)
                        if norm > 0:
                            features_array = features_array / norm
                            cleaned_data.append({
                                'features': features_array,
                                'quality': quality_list[i],
                                'position': face_positions[i]
                            })
                except:
                    continue

            if len(cleaned_data) >= 3:
                # Remove duplicates (similar features)
                unique_features = []
                unique_qualities = []
                unique_positions = []

                for data in cleaned_data:
                    feat = data['features']
                    is_duplicate = False
                    for existing_feat in unique_features:
                        similarity = self.compare_features(feat, existing_feat)
                        if similarity > 0.95:  # Very similar features
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        unique_features.append(feat)
                        unique_qualities.append(data['quality'])
                        unique_positions.append(data['position'])

                # Ensure this face isn't already registered under another name
                duplicate_face_found = any(
                    self.face_already_registered(features) for features in unique_features
                )

                if duplicate_face_found:
                    self.update_status(
                        "⛔ Ижил нүүр илэрсэн тул бүртгэл цуцлагдлаа.")
                    self.show_warning_async(
                        "Анхааруулга",
                        "Энэ нүүр аль хэдийн системд бүртгэлтэй байна."
                    )
                else:
                    for i, features in enumerate(unique_features):
                        self.save_embedding_to_db(
                            self.register_name, features, unique_qualities[i])

                    avg_quality = np.mean(unique_qualities)
                    self.update_status(
                        f"✅ {self.register_name} амжилттай бүртгэгдлээ!")
                    self.update_status(f"📊 Дундаж чанар: {avg_quality:.1f}%")
                    self.update_status(
                        f"🧹 Цэвэрлэсэн: {len(unique_features)}/{len(cleaned_data)} зураг")
                    self.update_status_display()
            else:
                self.update_status(f"❌ Хангалттай цэвэр дата байхгүй!")
        else:
            self.update_status(f"❌ Хангалттай зураг аваагүй!")

        self.stop_capture()

    def is_new_angle(self, center, positions, min_diff=30):
        """Check if position is new"""
        for pos in positions:
            dist = np.sqrt((center[0] - pos[0])**2 + (center[1] - pos[1])**2)
            if dist < min_diff:
                return False
        return True

    def start_recognition(self):
        """Start recognition"""
        if not self.known_face_names:
            messagebox.showwarning("Анхааруулга", "Эхлээд дата ачаална уу!")
            return

        if self.is_capturing:
            messagebox.showwarning("Анхааруулга", "Өөр үйлдэл явагдаж байна!")
            return

        self.current_mode = 'recognize'
        self.is_capturing = True
        self.register_btn.config(state='disabled')
        self.recognize_btn.config(state='disabled')
        self.stop_btn.config(state='normal')

        self.update_status("\n🎥 AI танилт эхэллээ...")
        self.info_label.config(text="🔍 Танилт явагдаж байна...")

        threading.Thread(target=self.recognize_thread, daemon=True).start()

    def recognize_thread(self):
        """Recognition thread"""
        self.video_capture = cv2.VideoCapture(0)

        frame_count = 0
        last_results = {}
        fps_start = time.time()
        fps_counter = 0

        while self.is_capturing:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            frame_count += 1
            fps_counter += 1

            if time.time() - fps_start >= 1.0:
                self.fps = fps_counter
                fps_counter = 0
                fps_start = time.time()

            if frame_count % 2 == 0:
                detections = self.detect_faces_mtcnn(frame, min_size=80)
                new_results = {}

                for face_id, face_info in enumerate(detections):
                    x, y, w, h = face_info['box']
                    face_roi = frame[y:y+h, x:x+w]

                    if face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                        continue

                    features = self.extract_facenet_embedding(face_roi)

                    if features is not None and len(features) > 0:
                        try:
                            features_array = np.array(
                                features, dtype=np.float32)
                            if not np.any(np.isnan(features_array)) and not np.any(np.isinf(features_array)):
                                name, confidence = self.find_best_match(
                                    features_array)
                                new_results[face_id] = (
                                    x, y, w, h, name, confidence)
                        except Exception:
                            continue

                last_results = new_results

            for face_id, (x, y, w, h, name, confidence) in last_results.items():
                color = self.get_color(name, confidence)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

                corner_len = 20
                for (cx, cy) in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
                    dx = corner_len if cx == x else -corner_len
                    dy = corner_len if cy == y else -corner_len
                    cv2.line(frame, (cx, cy), (cx+dx, cy), color, 5)
                    cv2.line(frame, (cx, cy), (cx, cy+dy), color, 5)

                if self.show_confidence_var.get() and name != "Танигдаагүй":
                    bar_width = int(w * (confidence / 100))
                    cv2.rectangle(frame, (x, y-10),
                                  (x+bar_width, y-5), color, -1)

                label_text = f"{name} ({confidence:.0f}%)" if name != "Танигдаагүй" else name
                label_y = y - 15 if y - 15 > 15 else y + h + 35

                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(frame, (x, label_y - text_height - 10),
                              (x + text_width + 10, label_y), color, -1)
                cv2.putText(frame, label_text, (x + 5, label_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            self.draw_hud(frame, len(last_results))
            self.display_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()
        self.stop_capture()

    def find_best_match(self, features):
        """Find best match"""
        if not self.known_face_features:
            return "Танигдаагүй", 0

        if features is None:
            return "Танигдаагүй", 0

        try:
            # Ensure input features are valid
            features = np.array(features, dtype=np.float32)
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return "Танигдаагүй", 0

            max_similarity = 0
            best_name = "Танигдаагүй"
            best_idx = -1

            # Compare with all known faces
            for idx, known_features in enumerate(self.known_face_features):
                if known_features is None:
                    continue

                similarity = self.compare_features(features, known_features)

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_name = self.known_face_names[idx]
                    best_idx = idx

            # Check if similarity meets threshold
            if max_similarity >= self.threshold:
                confidence = max_similarity * 100
                return best_name, confidence
            else:
                return "Танигдаагүй", max_similarity * 100
        except Exception as e:
            return "Танигдаагүй", 0

    def get_color(self, name, confidence):
        """Get color"""
        if name != "Танигдаагүй":
            if confidence > 80:
                return (0, 255, 159)
            elif confidence > 70:
                return (0, 191, 255)
            else:
                return (0, 165, 255)
        return (0, 0, 255)

    def draw_hud(self, frame, face_count):
        """Draw HUD"""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (26, 31, 58), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        info = f"FPS: {self.fps} | Faces: {face_count} | Enhanced OpenCV"
        cv2.putText(frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 159), 2)

        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_progress(self, frame, current, total):
        """Draw progress bar"""
        h, w = frame.shape[:2]
        bar_width = w - 80
        bar_height = 35
        bar_x, bar_y = 40, h - 60

        cv2.rectangle(frame, (bar_x-5, bar_y-5),
                      (bar_x + bar_width + 5, bar_y + bar_height + 5),
                      (26, 31, 58), -1)

        progress = int((current / total) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + progress, bar_y + bar_height),
                      (0, 255, 159), -1)

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + bar_height),
                      (255, 255, 255), 2)

        text = f"{current}/{total} ({int(current/total*100)}%)"
        cv2.putText(frame, text, (bar_x + bar_width//2 - 60, bar_y + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def stop_capture(self):
        """Stop capture"""
        self.is_capturing = False
        if self.video_capture:
            self.video_capture.release()

        self.register_btn.config(state='normal')
        self.recognize_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.info_label.config(text="⚡ Бэлэн")

        self.video_label.config(
            image='',
            text="🎥 Видео зогссон байна\n\n✨ Дахин эхлүүлэхийн тулд товч дарна уу"
        )

    def display_frame(self, frame):
        """Display frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((900, 650), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk, text='')

    def update_status(self, message, clear=False):
        """Update status"""
        self.status_text.config(state='normal')
        if clear:
            self.status_text.delete(1.0, tk.END)

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state='disabled')

    def update_status_display(self):
        """Update status display"""
        self.update_status("", clear=True)
        self.update_status("🟢 Enhanced OpenCV Mode")

        if self.known_face_names:
            name_counts = Counter(self.known_face_names)
            self.update_status(f"\n👥 Бүртгэлтэй: {len(name_counts)} хүн")
            self.update_status(f"📊 Нийт зураг: {len(self.known_face_names)}")

            if self.face_quality_scores:
                avg_quality = np.mean(self.face_quality_scores)
                self.update_status(f"✨ Дундаж чанар: {avg_quality:.1f}%")

            self.update_status(f"🎯 Threshold: {self.threshold:.2f}\n")
            self.update_status("📋 Хүмүүс:")

            for name, count in sorted(name_counts.items()):
                self.update_status(f"  • {name}: {count} зураг")
        else:
            self.update_status("\n⚠️ Бүртгэлтэй хүн байхгүй")

    def update_threshold(self, value):
        """Update threshold"""
        self.threshold = float(value)
        self.threshold_label.config(text=f"Утга: {self.threshold:.2f}")

    def save_data(self):
        """Save data with cleaning"""
        if not self.known_face_names:
            messagebox.showwarning("Анхааруулга", "Хадгалах дата байхгүй!")
            return

        try:
            data = {
                'features': [np.array(f, dtype=np.float32) for f in self.known_face_features],
                'names': self.known_face_names,
                'quality_scores': self.face_quality_scores,
                'threshold': self.threshold,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'version': 'facenet-mtcnn'
            }
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)

            self.update_status("💾 SQLite дата нөөц файлд хадгалагдлаа!")
            messagebox.showinfo(
                "Амжилт", f"Нөөц хадгаллаа!\n{len(set(self.known_face_names))} хүн\n{len(self.known_face_features)} embedding"
            )
        except Exception as e:
            messagebox.showerror("Алдаа", f"Хадгалах алдаа: {e}")

    def load_data_silent(self):
        """Load data silently with cleaning"""
        self.refresh_memory_from_db()
        if not self.known_face_names and os.path.exists(self.data_file):
            # Attempt automatic migration from legacy pickle
            self.import_legacy_pickle(silent=True)

    def load_data(self):
        """Load data with cleaning"""
        if not os.path.exists(self.data_file):
            messagebox.showwarning("Анхааруулга", "Нөөц файл олдсонгүй!")
            return

        self.import_legacy_pickle(silent=False)

    def import_legacy_pickle(self, silent=False):
        """Import embeddings from legacy pickle backup into SQLite"""
        if not os.path.exists(self.data_file):
            if not silent:
                messagebox.showwarning("Анхааруулга", "Нөөц файл олдсонгүй!")
            return

        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                raw_features = data.get('features', [])
                raw_names = data.get('names', [])
                raw_qualities = data.get('quality_scores', [])
                self.threshold = data.get('threshold', self.threshold)
                if hasattr(self, 'threshold_var'):
                    self.threshold_var.set(self.threshold)

            cleaned_features = []
            cleaned_names = []
            cleaned_qualities = []

            for i, features in enumerate(raw_features):
                try:
                    features_array = np.array(features, dtype=np.float32)
                    if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                        continue
                    norm = np.linalg.norm(features_array)
                    if norm > 0:
                        features_array = features_array / norm
                        cleaned_features.append(features_array)
                        cleaned_names.append(
                            raw_names[i] if i < len(raw_names) else "Unknown")
                        cleaned_qualities.append(
                            raw_qualities[i] if i < len(raw_qualities) else 50.0)
                except:
                    continue

            if cleaned_features:
                self.bulk_insert_embeddings(
                    cleaned_names, cleaned_features, cleaned_qualities)
                self.update_status_display()
                if not silent:
                    messagebox.showinfo(
                        "Амжилт",
                        f"Legacy pickle-ээс {len(cleaned_features)} embedding импортлолоо!"
                    )
            elif not silent:
                messagebox.showwarning(
                    "Анхааруулга", "Импортлох хүчинтэй embedding олдсонгүй!")
        except Exception as e:
            if not silent:
                messagebox.showerror("Алдаа", f"Импортын алдаа: {e}")

        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                raw_features = data.get('features', [])
                raw_names = data.get('names', [])
                raw_qualities = data.get('quality_scores', [])
                self.threshold = data.get('threshold', self.threshold)
                self.threshold_var.set(self.threshold)

            # Clean loaded data
            cleaned_features = []
            cleaned_names = []
            cleaned_qualities = []

            for i, features in enumerate(raw_features):
                try:
                    features_array = np.array(features, dtype=np.float32)

                    # Remove invalid features
                    if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                        continue

                    # Normalize
                    norm = np.linalg.norm(features_array)
                    if norm > 0:
                        features_array = features_array / norm
                        cleaned_features.append(features_array)
                        cleaned_names.append(
                            raw_names[i] if i < len(raw_names) else "Unknown")
                        cleaned_qualities.append(
                            raw_qualities[i] if i < len(raw_qualities) else 50.0)
                except:
                    continue

            # Update with cleaned data
            self.known_face_features = cleaned_features
            self.known_face_names = cleaned_names
            self.face_quality_scores = cleaned_qualities

            self.update_status_display()
            cleaned_count = len(cleaned_features)
            original_count = len(raw_features)
            if cleaned_count < original_count:
                messagebox.showinfo("Амжилт",
                                    f"Ачаалагдлаа!\n{len(set(cleaned_names))} хүн\n"
                                    f"🧹 Цэвэрлэсэн: {cleaned_count}/{original_count} зураг")
            else:
                messagebox.showinfo(
                    "Амжилт", f"Ачаалагдлаа!\n{len(set(cleaned_names))} хүн")
        except Exception as e:
            messagebox.showerror("Алдаа", f"Ачаалах алдаа: {e}")

    def export_json(self):
        """Export to JSON"""
        if not self.known_face_names:
            messagebox.showwarning("Анхааруулга", "Экспорт хийх дата байхгүй!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                name_counts = Counter(self.known_face_names)
                export_data = {
                    'people': [
                        {
                            'name': name,
                            'sample_count': count,
                            'avg_quality': float(np.mean([
                                self.face_quality_scores[i]
                                for i, n in enumerate(self.known_face_names) if n == name
                            ]))
                        }
                        for name, count in name_counts.items()
                    ],
                    'total_samples': len(self.known_face_names),
                    'threshold': self.threshold,
                    'export_date': datetime.now().isoformat()
                }

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

                self.update_status(f"📤 Export: {filename}")
                messagebox.showinfo("Амжилт", "JSON амжилттай!")
            except Exception as e:
                messagebox.showerror("Алдаа", f"Export алдаа: {e}")

    def import_from_folder(self):
        """Import from folder"""
        folder = filedialog.askdirectory(title="Зургийн фолдер сонгох")

        if folder:
            files = [f for f in os.listdir(folder)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            if not files:
                messagebox.showwarning("Анхааруулга", "Зураг олдсонгүй!")
                return

            success = 0
            for filename in files:
                path = os.path.join(folder, filename)
                image = cv2.imread(path)

                if image is None:
                    continue

                faces = self.detect_faces_mtcnn(image, min_size=80)

                if len(faces) > 0:
                    face_info = max(
                        faces, key=lambda info: info['box'][2] * info['box'][3])
                    x, y, w, h = face_info['box']
                    face_roi = image[y:y+h, x:x+w]

                    features = self.extract_facenet_embedding(face_roi)

                    if features is not None:
                        name = os.path.splitext(
                            filename)[0].replace('_', ' ').title()
                        quality = self.calculate_face_quality(face_roi)

                        self.save_embedding_to_db(name, features, quality)
                        success += 1

            if success > 0:
                self.update_status_display()
                messagebox.showinfo(
                    "Амжилт", f"{success}/{len(files)} зураг импортлогдлоо!")
            else:
                messagebox.showwarning(
                    "Анхааруулга", "Амжилттай импортлосон зураг байхгүй!")

    def show_people_list(self):
        """Show people list"""
        if not self.known_face_names:
            messagebox.showinfo("Мэдээлэл", "Бүртгэлтэй хүн байхгүй")
            return

        name_counts = Counter(self.known_face_names)
        message = "📋 БҮРТГЭЛТЭЙ ХҮМҮҮС\n" + "="*40 + "\n\n"

        for name, count in sorted(name_counts.items()):
            qualities = [self.face_quality_scores[i]
                         for i, n in enumerate(self.known_face_names) if n == name]
            avg_q = np.mean(qualities) if qualities else 0

            message += f"👤 {name}\n"
            message += f"   📊 Зураг: {count}\n"
            message += f"   ✨ Чанар: {avg_q:.1f}%\n\n"

        messagebox.showinfo("Бүртгэлтэй хүмүүс", message)

    def show_statistics(self):
        """Show statistics"""
        if not self.known_face_names:
            messagebox.showinfo("Мэдээлэл", "Статистик байхгүй")
            return

        name_counts = Counter(self.known_face_names)
        total_people = len(name_counts)
        total_samples = len(self.known_face_names)
        avg_quality = np.mean(
            self.face_quality_scores) if self.face_quality_scores else 0

        message = "📊 СТАТИСТИК\n" + "="*40 + "\n\n"
        message += f"👥 Нийт хүн: {total_people}\n"
        message += f"📸 Нийт зураг: {total_samples}\n"
        message += f"✨ Дундаж чанар: {avg_quality:.1f}%\n"
        message += f"🎯 Threshold: {self.threshold:.2f}\n\n"

        message += "📈 Хүн бүрийн зураг:\n"
        for name, count in name_counts.most_common():
            message += f"  • {name}: {count}\n"

        messagebox.showinfo("Статистик", message)

    def delete_person(self):
        """Delete person"""
        if not self.known_face_names:
            messagebox.showinfo("Мэдээлэл", "Бүртгэлтэй хүн байхгүй")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("🗑️ Хүн устгах")
        dialog.geometry("450x350")
        dialog.configure(bg=self.bg_panel)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="⚠️ Устгах хүнийг сонгоно уу:",
                 font=self.get_font(12, 'bold'),
                 bg=self.bg_panel, fg=self.fg_secondary).pack(pady=20)

        name_counts = Counter(self.known_face_names)
        names = sorted(name_counts.keys())

        listbox = tk.Listbox(dialog, font=self.get_font(11), height=10,
                             bg='#2a2f4a', fg=self.fg_secondary,
                             selectbackground=self.fg_primary,
                             selectforeground=self.bg_dark)
        listbox.pack(fill='both', expand=True, padx=20, pady=10)

        for name in names:
            listbox.insert(tk.END, f"{name} ({name_counts[name]} зураг)")

        def delete_selected():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("Анхааруулга", "Хүн сонгоно уу!")
                return

            name = names[selection[0]]
            confirm = messagebox.askyesno(
                "Баталгаажуулалт",
                f"'{name}' устгах уу?\n\nБуцаах боломжгүй!"
            )

            if confirm:
                self.delete_person_from_db(name)
                self.update_status(f"🗑️ {name} устгагдлаа!")
                self.update_status_display()
                dialog.destroy()

        delete_btn = tk.Button(dialog, text="🗑️ Устгах", command=delete_selected,
                               bg='#ff4466', fg='#ffffff',
                               font=self.get_font(11, 'bold'),
                               cursor='hand2', height=2, relief='flat',
                               activebackground=self.lighten_color('#ff4466'),
                               activeforeground='#ffffff')
        delete_btn.pack(pady=10)


def main():
    root = tk.Tk()
    app = EnhancedFaceRecognitionSystem(root)
    root.mainloop()


if __name__ == "__main__":
    print("="*60)
    print("🚀 ENHANCED FACE RECOGNITION - OpenCV Only")
    print("="*60)
    print("✅ dlib шаардлагагүй")
    print("✅ Deep features: LBP + HOG + ORB + Histogram")
    print("✅ Quality scoring")
    print("✅ Multi-angle capture")
    print("="*60)
    print("\nПрограм эхэлж байна...\n")

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Програм зогссон")
    except Exception as e:
        print(f"\n❌ Алдаа: {e}")
        import traceback
        traceback.print_exc()
