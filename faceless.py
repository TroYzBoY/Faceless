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


class EnhancedFaceRecognitionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 AI Нүүр таних систем - Enhanced")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0a0e27')
        
        # Face data storage
        self.known_face_features = []
        self.known_face_names = []
        self.face_quality_scores = []
        self.data_file = "enhanced_face_data.pkl"
        self.threshold = 0.68
        
        # Initialize OpenCV
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_eye.xml')
        
        # Video capture
        self.video_capture = None
        self.is_capturing = False
        self.current_mode = None
        self.fps = 0
        
        self.setup_ui()
        self.load_data_silent()
    
    def setup_ui(self):
        """Setup modern UI"""
        # Title bar
        title_frame = tk.Frame(self.root, bg='#1a1f3a', height=90)
        title_frame.pack(fill='x', pady=(0, 10))
        
        title_label = tk.Label(
            title_frame, 
            text="🚀 AI НҮҮР ТАНИХ СИСТЕМ", 
            font=('Segoe UI', 26, 'bold'),
            bg='#1a1f3a',
            fg='#00ff9f'
        )
        title_label.pack(pady=15)
        
        mode_label = tk.Label(
            title_frame,
            text="🟢 ENHANCED MODE (OpenCV + Deep Features)",
            font=('Segoe UI', 10, 'bold'),
            bg='#1a1f3a',
            fg='#00ff9f'
        )
        mode_label.pack()
        
        # Main container
        main_container = tk.Frame(self.root, bg='#0a0e27')
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel
        left_panel = tk.Frame(main_container, bg='#1a1f3a', width=380)
        left_panel.pack(side='left', fill='both', padx=(0, 10))
        
        # Control buttons
        control_frame = tk.LabelFrame(
            left_panel, 
            text="⚡ Үндсэн үйлдлүүд", 
            font=('Segoe UI', 12, 'bold'),
            bg='#1a1f3a',
            fg='#ffffff',
            padx=15,
            pady=15
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
            font=('Segoe UI', 12, 'bold'),
            bg='#1a1f3a',
            fg='#ffffff',
            padx=15,
            pady=15
        )
        advanced_frame.pack(fill='x', pady=10, padx=10)
        
        self.multi_angle_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            advanced_frame, text="📐 Олон өнцгөөс авах",
            variable=self.multi_angle_var, bg='#1a1f3a', fg='#ffffff',
            selectcolor='#2a2f4a', font=('Segoe UI', 10)
        ).pack(anchor='w', pady=3)
        
        self.quality_filter_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            advanced_frame, text="✨ Чанарын шүүлтүүр",
            variable=self.quality_filter_var, bg='#1a1f3a', fg='#ffffff',
            selectcolor='#2a2f4a', font=('Segoe UI', 10)
        ).pack(anchor='w', pady=3)
        
        self.deep_features_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            advanced_frame, text="🧠 Deep features (LBP+HOG+ORB)",
            variable=self.deep_features_var, bg='#1a1f3a', fg='#ffffff',
            selectcolor='#2a2f4a', font=('Segoe UI', 10)
        ).pack(anchor='w', pady=3)
        
        self.show_confidence_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            advanced_frame, text="📊 Confidence bar",
            variable=self.show_confidence_var, bg='#1a1f3a', fg='#ffffff',
            selectcolor='#2a2f4a', font=('Segoe UI', 10)
        ).pack(anchor='w', pady=3)
        
        # Data management
        data_frame = tk.LabelFrame(
            left_panel, text="💾 Дата удирдлага", 
            font=('Segoe UI', 12, 'bold'),
            bg='#1a1f3a', fg='#ffffff', padx=15, pady=15
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
            self.create_button(data_frame, text, cmd, color).pack(fill='x', pady=3)
        
        # Settings
        settings_frame = tk.LabelFrame(
            left_panel, text="⚙️ Тохиргоо", 
            font=('Segoe UI', 12, 'bold'),
            bg='#1a1f3a', fg='#ffffff', padx=15, pady=15
        )
        settings_frame.pack(fill='x', pady=10, padx=10)
        
        tk.Label(settings_frame, text="Threshold утга:", 
                bg='#1a1f3a', fg='#ffffff', font=('Segoe UI', 10)).pack(anchor='w')
        
        self.threshold_var = tk.DoubleVar(value=self.threshold)
        threshold_slider = ttk.Scale(
            settings_frame, from_=0.50, to=0.85,
            variable=self.threshold_var, orient='horizontal',
            command=self.update_threshold
        )
        threshold_slider.pack(fill='x', pady=5)
        
        self.threshold_label = tk.Label(
            settings_frame, text=f"Утга: {self.threshold:.2f}",
            bg='#1a1f3a', fg='#00ff9f', font=('Segoe UI', 9, 'bold')
        )
        self.threshold_label.pack()
        
        # Status display
        status_frame = tk.LabelFrame(
            left_panel, text="📊 Систем мэдээлэл", 
            font=('Segoe UI', 12, 'bold'),
            bg='#1a1f3a', fg='#ffffff', padx=15, pady=15
        )
        status_frame.pack(fill='both', expand=True, pady=10, padx=10)
        
        self.status_text = tk.Text(
            status_frame, height=12, bg='#0a0e27', fg='#00ff9f',
            font=('Consolas', 9), wrap='word', state='disabled',
            borderwidth=0, highlightthickness=0
        )
        self.status_text.pack(fill='both', expand=True)
        
        # Right panel - Video
        right_panel = tk.Frame(main_container, bg='#1a1f3a')
        right_panel.pack(side='right', fill='both', expand=True)
        
        video_frame = tk.LabelFrame(
            right_panel, text="📹 Видео харагдац",
            font=('Segoe UI', 12, 'bold'),
            bg='#1a1f3a', fg='#ffffff'
        )
        video_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.video_label = tk.Label(
            video_frame, bg='#0a0e27',
            text="🎥 Видео зогссон байна\n\n✨ Эхлүүлэхийн тулд дээрх товчийг дарна уу",
            font=('Segoe UI', 14), fg='#666699'
        )
        self.video_label.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Info bar
        info_frame = tk.Frame(right_panel, bg='#1a1f3a', height=40)
        info_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.info_label = tk.Label(
            info_frame, text="⚡ Бэлэн",
            font=('Segoe UI', 10), bg='#1a1f3a', fg='#00ff9f'
        )
        self.info_label.pack(side='left', padx=10, pady=5)
        
        self.update_status_display()
    
    def create_button(self, parent, text, command, color):
        """Create styled button"""
        btn = tk.Button(
            parent, text=text, command=command,
            bg=color, fg='#ffffff', font=('Segoe UI', 11, 'bold'),
            relief='flat', cursor='hand2', height=2,
            activebackground=self.lighten_color(color), borderwidth=0
        )
        btn.bind('<Enter>', lambda e: btn.config(bg=self.lighten_color(color)))
        btn.bind('<Leave>', lambda e: btn.config(bg=color))
        return btn
    
    def lighten_color(self, color):
        """Lighten hex color"""
        colors = {
            '#00ff9f': '#33ffb3', '#00aaff': '#33bbff',
            '#ff4466': '#ff6688', '#9966ff': '#aa77ff',
            '#ff9500': '#ffaa33'
        }
        return colors.get(color, color)
    
    def extract_deep_features(self, face_image):
        """Extract deep features: LBP + HOG + ORB + Histogram"""
        try:
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            gray = cv2.resize(gray, (128, 128))
            gray = cv2.equalizeHist(gray)
            
            # 1. Histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()[:64]
            
            # 2. LBP
            lbp = self.compute_lbp(gray)
            lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()[:64]
            
            # 3. HOG
            hog = self.compute_hog(gray)[:128]
            
            # 4. ORB
            orb_feat = self.compute_orb_features(gray)
            
            # Combine
            combined = np.concatenate([hist, lbp_hist, hog, orb_feat])
            
            # Normalize
            if np.linalg.norm(combined) > 0:
                combined = combined / np.linalg.norm(combined)
            
            return combined
        except Exception as e:
            return None
    
    def compute_lbp(self, image):
        """Local Binary Pattern"""
        height, width = image.shape
        lbp = np.zeros((height-2, width-2), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] >= center) << 7
                code |= (image[i-1, j] >= center) << 6
                code |= (image[i-1, j+1] >= center) << 5
                code |= (image[i, j+1] >= center) << 4
                code |= (image[i+1, j+1] >= center) << 3
                code |= (image[i+1, j] >= center) << 2
                code |= (image[i+1, j-1] >= center) << 1
                code |= (image[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        
        return lbp
    
    def compute_hog(self, image):
        """HOG features"""
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        bins = np.int32(angle / 40) % 9
        hist = []
        
        cell_size = 16
        for i in range(0, image.shape[0] - cell_size, cell_size):
            for j in range(0, image.shape[1] - cell_size, cell_size):
                cell_mag = mag[i:i+cell_size, j:j+cell_size]
                cell_bins = bins[i:i+cell_size, j:j+cell_size]
                
                cell_hist = np.zeros(9)
                for k in range(9):
                    cell_hist[k] = np.sum(cell_mag[cell_bins == k])
                
                hist.extend(cell_hist)
        
        hog_features = np.array(hist)
        if np.linalg.norm(hog_features) > 0:
            hog_features = hog_features / np.linalg.norm(hog_features)
        
        return hog_features
    
    def compute_orb_features(self, image):
        """ORB features"""
        try:
            orb = cv2.ORB_create(nfeatures=50)
            keypoints, descriptors = orb.detectAndCompute(image, None)
            
            if descriptors is not None and len(descriptors) > 0:
                avg_desc = np.mean(descriptors, axis=0)
                if len(avg_desc) > 32:
                    return avg_desc[:32]
                else:
                    padded = np.zeros(32)
                    padded[:len(avg_desc)] = avg_desc
                    return padded
            else:
                return np.zeros(32)
        except:
            return np.zeros(32)
    
    def calculate_face_quality(self, face_image):
        """Calculate quality score"""
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            brightness = np.mean(gray)
            brightness_score = 100 - abs(brightness - 128)
            
            contrast = gray.std()
            
            quality = min(100, (sharpness * 3 + brightness_score + contrast) / 5)
            return max(0, quality)
        except:
            return 50.0
    
    def compare_features(self, feat1, feat2):
        """Compare features"""
        cos_sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-6)
        euclidean_dist = np.linalg.norm(feat1 - feat2)
        euclidean_sim = 1 / (1 + euclidean_dist)
        similarity = 0.7 * cos_sim + 0.3 * euclidean_sim
        return similarity
    
    def start_registration(self):
        """Start registration"""
        if self.is_capturing:
            messagebox.showwarning("Анхааруулга", "Өөр үйлдэл явагдаж байна!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("✨ Нүүр бүртгэх")
        dialog.geometry("450x280")
        dialog.configure(bg='#1a1f3a')
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="👤 Хүний нэр оруулна уу:",
                font=('Segoe UI', 13, 'bold'),
                bg='#1a1f3a', fg='#ffffff').pack(pady=25)
        
        name_entry = tk.Entry(dialog, font=('Segoe UI', 12), width=30,
                             bg='#2a2f4a', fg='#ffffff',
                             insertbackground='#00ff9f', relief='flat', borderwidth=5)
        name_entry.pack(pady=10)
        name_entry.focus()
        
        tk.Label(dialog, text="📸 Зургийн тоо:",
                font=('Segoe UI', 10),
                bg='#1a1f3a', fg='#ffffff').pack(pady=(10, 5))
        
        sample_var = tk.IntVar(value=15)
        tk.Spinbox(dialog, from_=8, to=25, textvariable=sample_var,
                  font=('Segoe UI', 11), width=10,
                  bg='#2a2f4a', fg='#ffffff').pack()
        
        def submit():
            name = name_entry.get().strip()
            if name:
                dialog.destroy()
                self.register_name = name
                self.register_samples = sample_var.get()
                threading.Thread(target=self.register_thread, daemon=True).start()
            else:
                messagebox.showerror("Алдаа", "Нэр оруулна уу!")
        
        tk.Button(dialog, text="✓ Эхлүүлэх", command=submit,
                 bg='#00ff9f', fg='#0a0e27',
                 font=('Segoe UI', 11, 'bold'),
                 cursor='hand2', height=2, relief='flat').pack(pady=15)
        
        name_entry.bind('<Return>', lambda e: submit())
    
    def register_thread(self):
        """Registration thread"""
        self.is_capturing = True
        self.current_mode = 'register'
        self.register_btn.config(state='disabled')
        self.recognize_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        self.update_status(f"\n🚀 {self.register_name} бүртгэж байна...")
        self.info_label.config(text=f"📸 Бүртгэл: {self.register_name}")
        
        self.video_capture = cv2.VideoCapture(0)
        
        features_list = []
        quality_list = []
        count = 0
        face_positions = []
        last_capture = time.time()
        stable_frames = 0
        
        while count < self.register_samples and self.is_capturing:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(120, 120), maxSize=(400, 400)
            )
            
            current_time = time.time()
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, minNeighbors=8)
                has_eyes = len(eyes) >= 2
                
                face_center = (x + w//2, y + h//2)
                is_new_angle = self.is_new_angle(face_center, face_positions) if self.multi_angle_var.get() else True
                
                face_roi = frame[y:y+h, x:x+w]
                quality = self.calculate_face_quality(face_roi)
                quality_ok = quality > 35 if self.quality_filter_var.get() else True
                
                ready = has_eyes and is_new_angle and quality_ok
                
                if ready:
                    color = (0, 255, 0)
                    stable_frames += 1
                else:
                    if not quality_ok:
                        color = (255, 0, 255)
                    elif has_eyes:
                        color = (0, 255, 255)
                    else:
                        color = (0, 165, 255)
                    stable_frames = 0
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                if self.quality_filter_var.get():
                    cv2.putText(frame, f"Q: {quality:.0f}%", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if stable_frames >= 3 and current_time - last_capture >= 0.5:
                    if self.deep_features_var.get():
                        features = self.extract_deep_features(face_roi)
                    else:
                        features = self.extract_simple_features(face_roi)
                    
                    if features is not None:
                        features_list.append(features)
                        quality_list.append(quality)
                        face_positions.append(face_center)
                        count += 1
                        last_capture = current_time
                        stable_frames = 0
                        
                        overlay = frame.copy()
                        cv2.circle(overlay, (frame.shape[1]//2, frame.shape[0]//2), 80, (0, 255, 0), -1)
                        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                        
                        self.update_status(f"📸 {count}/{self.register_samples} - Q: {quality:.0f}%")
            
            self.draw_progress(frame, count, self.register_samples)
            self.display_frame(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.video_capture.release()
        
        if count >= 3:
            for i, features in enumerate(features_list):
                self.known_face_features.append(features)
                self.known_face_names.append(self.register_name)
                self.face_quality_scores.append(quality_list[i])
            
            avg_quality = np.mean(quality_list)
            self.update_status(f"✅ {self.register_name} амжилттай бүртгэгдлээ!")
            self.update_status(f"📊 Дундаж чанар: {avg_quality:.1f}%")
            self.save_data()
            self.update_status_display()
        else:
            self.update_status(f"❌ Хангалттай зураг аваагүй!")
        
        self.stop_capture()
    
    def extract_simple_features(self, face_image):
        """Simple histogram features"""
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            gray = cv2.resize(gray, (100, 100))
            gray = cv2.equalizeHist(gray)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            return cv2.normalize(hist, hist).flatten()
        except:
            return None
    
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
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5,
                    minSize=(60, 60), maxSize=(400, 400)
                )
                
                new_results = {}
                
                for face_id, (x, y, w, h) in enumerate(faces):
                    face_roi = frame[y:y+h, x:x+w]
                    
                    if self.deep_features_var.get():
                        features = self.extract_deep_features(face_roi)
                    else:
                        features = self.extract_simple_features(face_roi)
                    
                    if features is not None:
                        name, confidence = self.find_best_match(features)
                        new_results[face_id] = (x, y, w, h, name, confidence)
                
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
                    cv2.rectangle(frame, (x, y-10), (x+bar_width, y-5), color, -1)
                
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
        
        max_similarity = 0
        best_name = "Танигдаагүй"
        
        for idx, known_features in enumerate(self.known_face_features):
            similarity = self.compare_features(features, known_features)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_name = self.known_face_names[idx]
        
        if max_similarity > self.threshold:
            return best_name, max_similarity * 100
        else:
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
        """Save data"""
        if not self.known_face_names:
            messagebox.showwarning("Анхааруулга", "Хадгалах дата байхгүй!")
            return
        
        try:
            data = {
                'features': self.known_face_features,
                'names': self.known_face_names,
                'quality_scores': self.face_quality_scores,
                'threshold': self.threshold,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'version': '2.0-opencv'
            }
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.update_status("💾 Дата хадгалагдлаа!")
            messagebox.showinfo("Амжилт", f"Амжилттай!\n{len(set(self.known_face_names))} хүн")
        except Exception as e:
            messagebox.showerror("Алдаа", f"Хадгалах алдаа: {e}")
    
    def load_data_silent(self):
        """Load data silently"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_features = data.get('features', [])
                    self.known_face_names = data.get('names', [])
                    self.face_quality_scores = data.get('quality_scores', [])
                    self.threshold = data.get('threshold', self.threshold)
                    self.threshold_var.set(self.threshold)
            except:
                pass
    
    def load_data(self):
        """Load data"""
        if not os.path.exists(self.data_file):
            messagebox.showwarning("Анхааруулга", "Дата файл олдсонгүй!")
            return
        
        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_features = data.get('features', [])
                self.known_face_names = data.get('names', [])
                self.face_quality_scores = data.get('quality_scores', [])
                self.threshold = data.get('threshold', self.threshold)
                self.threshold_var.set(self.threshold)
            
            self.update_status_display()
            messagebox.showinfo("Амжилт", f"Ачаалагдлаа!\n{len(set(self.known_face_names))} хүн")
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
                
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                
                if len(faces) > 0:
                    face = max(faces, key=lambda r: r[2] * r[3])
                    x, y, w, h = face
                    face_roi = image[y:y+h, x:x+w]
                    
                    features = self.extract_deep_features(face_roi) if self.deep_features_var.get() else self.extract_simple_features(face_roi)
                    
                    if features is not None:
                        name = os.path.splitext(filename)[0].replace('_', ' ').title()
                        quality = self.calculate_face_quality(face_roi)
                        
                        self.known_face_features.append(features)
                        self.known_face_names.append(name)
                        self.face_quality_scores.append(quality)
                        success += 1
            
            if success > 0:
                self.save_data()
                self.update_status_display()
                messagebox.showinfo("Амжилт", f"{success}/{len(files)} зураг импортлогдлоо!")
            else:
                messagebox.showwarning("Анхааруулга", "Амжилттай импортлосон зураг байхгүй!")
    
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
        avg_quality = np.mean(self.face_quality_scores) if self.face_quality_scores else 0
        
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
        dialog.configure(bg='#1a1f3a')
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="⚠️ Устгах хүнийг сонгоно уу:",
                font=('Segoe UI', 12, 'bold'),
                bg='#1a1f3a', fg='#ffffff').pack(pady=20)
        
        name_counts = Counter(self.known_face_names)
        names = sorted(name_counts.keys())
        
        listbox = tk.Listbox(dialog, font=('Segoe UI', 11), height=10,
                            bg='#2a2f4a', fg='#ffffff',
                            selectbackground='#00ff9f',
                            selectforeground='#0a0e27')
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
                indices = [i for i, n in enumerate(self.known_face_names) if n == name]
                for idx in sorted(indices, reverse=True):
                    del self.known_face_features[idx]
                    del self.known_face_names[idx]
                    if idx < len(self.face_quality_scores):
                        del self.face_quality_scores[idx]
                
                self.update_status(f"🗑️ {name} устгагдлаа!")
                self.save_data()
                self.update_status_display()
                dialog.destroy()
        
        tk.Button(dialog, text="🗑️ Устгах", command=delete_selected,
                 bg='#ff4466', fg='#ffffff',
                 font=('Segoe UI', 11, 'bold'),
                 cursor='hand2', height=2, relief='flat').pack(pady=10)


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