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


class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Нүүр таних систем")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e2e')
        
        # Initialize face recognition system
        self.known_face_features = []
        self.known_face_names = []
        self.data_file = "face_data.pkl"
        self.threshold = 0.72
        
        # OpenCV cascades
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_eye.xml')
        
        # Video capture variables
        self.video_capture = None
        self.is_capturing = False
        self.current_mode = None  # 'register' or 'recognize'
        
        self.setup_ui()
        self.load_data_silent()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title bar
        title_frame = tk.Frame(self.root, bg='#2d2d44', height=80)
        title_frame.pack(fill='x', pady=(0, 10))
        
        title_label = tk.Label(
            title_frame, 
            text="📱 AUTO FACE ID СИСТЕМ", 
            font=('Helvetica', 24, 'bold'),
            bg='#2d2d44',
            fg='#00ff88'
        )
        title_label.pack(pady=20)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#1e1e2e')
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_container, bg='#2d2d44', width=350)
        left_panel.pack(side='left', fill='both', padx=(0, 10))
        
        # Control buttons
        control_frame = tk.LabelFrame(
            left_panel, 
            text="⚙️ Үндсэн үйлдлүүд", 
            font=('Helvetica', 12, 'bold'),
            bg='#2d2d44',
            fg='#ffffff',
            padx=15,
            pady=15
        )
        control_frame.pack(fill='x', pady=10, padx=10)
        
        # Register button
        self.register_btn = self.create_button(
            control_frame, 
            "🤖 Нүүр бүртгэх", 
            self.start_registration,
            '#00ff88'
        )
        self.register_btn.pack(fill='x', pady=5)
        
        # Recognize button
        self.recognize_btn = self.create_button(
            control_frame, 
            "🎥 Танилт эхлүүлэх", 
            self.start_recognition,
            '#00aaff'
        )
        self.recognize_btn.pack(fill='x', pady=5)
        
        # Stop button
        self.stop_btn = self.create_button(
            control_frame, 
            "⏹️ Зогсоох", 
            self.stop_capture,
            '#ff4444'
        )
        self.stop_btn.pack(fill='x', pady=5)
        self.stop_btn.config(state='disabled')
        
        # Data management
        data_frame = tk.LabelFrame(
            left_panel, 
            text="💾 Дата удирдлага", 
            font=('Helvetica', 12, 'bold'),
            bg='#2d2d44',
            fg='#ffffff',
            padx=15,
            pady=15
        )
        data_frame.pack(fill='x', pady=10, padx=10)
        
        self.create_button(
            data_frame, 
            "📂 Дата ачаалах", 
            self.load_data,
            '#9966ff'
        ).pack(fill='x', pady=5)
        
        self.create_button(
            data_frame, 
            "💾 Дата хадгалах", 
            self.save_data,
            '#9966ff'
        ).pack(fill='x', pady=5)
        
        self.create_button(
            data_frame, 
            "👥 Хүмүүсийг харах", 
            self.show_people_list,
            '#ff9500'
        ).pack(fill='x', pady=5)
        
        self.create_button(
            data_frame, 
            "🗑️ Хүн устгах", 
            self.delete_person,
            '#ff4444'
        ).pack(fill='x', pady=5)
        
        # Settings
        settings_frame = tk.LabelFrame(
            left_panel, 
            text="⚙️ Тохиргоо", 
            font=('Helvetica', 12, 'bold'),
            bg='#2d2d44',
            fg='#ffffff',
            padx=15,
            pady=15
        )
        settings_frame.pack(fill='x', pady=10, padx=10)
        
        # Threshold slider
        tk.Label(
            settings_frame, 
            text="Threshold:", 
            bg='#2d2d44', 
            fg='#ffffff',
            font=('Helvetica', 10)
        ).pack(anchor='w')
        
        self.threshold_var = tk.DoubleVar(value=self.threshold)
        threshold_slider = ttk.Scale(
            settings_frame,
            from_=0.70,
            to=0.95,
            variable=self.threshold_var,
            orient='horizontal',
            command=self.update_threshold
        )
        threshold_slider.pack(fill='x', pady=5)
        
        self.threshold_label = tk.Label(
            settings_frame,
            text=f"Утга: {self.threshold:.2f}",
            bg='#2d2d44',
            fg='#00ff88',
            font=('Helvetica', 9)
        )
        self.threshold_label.pack()
        
        # Status info
        status_frame = tk.LabelFrame(
            left_panel, 
            text="📊 Мэдээлэл", 
            font=('Helvetica', 12, 'bold'),
            bg='#2d2d44',
            fg='#ffffff',
            padx=15,
            pady=15
        )
        status_frame.pack(fill='both', expand=True, pady=10, padx=10)
        
        self.status_text = tk.Text(
            status_frame,
            height=10,
            bg='#1e1e2e',
            fg='#ffffff',
            font=('Courier', 9),
            wrap='word',
            state='disabled'
        )
        self.status_text.pack(fill='both', expand=True)
        
        # Right panel - Video feed
        right_panel = tk.Frame(main_container, bg='#2d2d44')
        right_panel.pack(side='right', fill='both', expand=True)
        
        video_label_frame = tk.LabelFrame(
            right_panel,
            text="📹 Видео",
            font=('Helvetica', 12, 'bold'),
            bg='#2d2d44',
            fg='#ffffff'
        )
        video_label_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.video_label = tk.Label(
            video_label_frame,
            bg='#1e1e2e',
            text="Видео зогссон байна\n\n🎥 'Нүүр бүртгэх' эсвэл 'Танилт эхлүүлэх' дарна уу",
            font=('Helvetica', 14),
            fg='#666666'
        )
        self.video_label.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.update_status_display()
    
    def create_button(self, parent, text, command, color):
        """Create a styled button"""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg='#ffffff',
            font=('Helvetica', 11, 'bold'),
            relief='flat',
            cursor='hand2',
            height=2,
            activebackground=self.lighten_color(color)
        )
        return btn
    
    def lighten_color(self, color):
        """Lighten a hex color"""
        # Simple color lightening
        if color == '#00ff88':
            return '#33ff99'
        elif color == '#00aaff':
            return '#33bbff'
        elif color == '#ff4444':
            return '#ff6666'
        elif color == '#9966ff':
            return '#aa77ff'
        elif color == '#ff9500':
            return '#ffaa33'
        return color
    
    def update_status(self, message, clear=False):
        """Update status text"""
        self.status_text.config(state='normal')
        if clear:
            self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state='disabled')
    
    def update_status_display(self):
        """Update the status information"""
        self.update_status("", clear=True)
        if self.known_face_names:
            name_counts = Counter(self.known_face_names)
            self.update_status(f"👥 Бүртгэлтэй: {len(name_counts)} хүн")
            self.update_status(f"📊 Нийт зураг: {len(self.known_face_names)}")
            self.update_status(f"🎯 Threshold: {self.threshold:.2f}\n")
            self.update_status("Хүмүүс:")
            for name, count in sorted(name_counts.items()):
                self.update_status(f"  • {name}: {count} зураг")
        else:
            self.update_status("⚠️ Бүртгэлтэй хүн байхгүй")
    
    def update_threshold(self, value):
        """Update threshold value"""
        self.threshold = float(value)
        self.threshold_label.config(text=f"Утга: {self.threshold:.2f}")
    
    def start_registration(self):
        """Start face registration process"""
        if self.is_capturing:
            messagebox.showwarning("Анхааруулга", "Өөр үйлдэл явагдаж байна!")
            return
        
        # Get name dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Нэр оруулах")
        dialog.geometry("400x200")
        dialog.configure(bg='#2d2d44')
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(
            dialog,
            text="Хүний нэр оруулна уу:",
            font=('Helvetica', 12),
            bg='#2d2d44',
            fg='#ffffff'
        ).pack(pady=20)
        
        name_entry = tk.Entry(
            dialog,
            font=('Helvetica', 12),
            width=30
        )
        name_entry.pack(pady=10)
        name_entry.focus()
        
        def submit():
            name = name_entry.get().strip()
            if name:
                dialog.destroy()
                self.current_mode = 'register'
                self.register_name = name
                self.register_samples = 10
                threading.Thread(target=self.register_face_thread, daemon=True).start()
            else:
                messagebox.showerror("Алдаа", "Нэр оруулна уу!")
        
        tk.Button(
            dialog,
            text="✓ Эхлүүлэх",
            command=submit,
            bg='#00ff88',
            fg='#ffffff',
            font=('Helvetica', 11, 'bold'),
            cursor='hand2',
            height=2
        ).pack(pady=10)
        
        name_entry.bind('<Return>', lambda e: submit())
    
    def register_face_thread(self):
        """Register face in separate thread"""
        self.is_capturing = True
        self.register_btn.config(state='disabled')
        self.recognize_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        self.update_status(f"\n📱 {self.register_name} бүртгэж байна...")
        
        self.video_capture = cv2.VideoCapture(0)
        
        features_list = []
        count = 0
        face_positions = []
        last_capture_time = time.time()
        stable_frames = 0
        
        while count < self.register_samples and self.is_capturing:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(100, 100), maxSize=(400, 400)
            )
            
            current_time = time.time()
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, minNeighbors=8)
                has_eyes = len(eyes) >= 2
                
                face_center = (x + w//2, y + h//2)
                is_new_angle = self.is_new_angle(face_center, face_positions)
                
                if has_eyes and is_new_angle:
                    color = (0, 255, 0)
                    stable_frames += 1
                    ready = stable_frames >= 3
                else:
                    color = (0, 255, 255) if has_eyes else (0, 165, 255)
                    stable_frames = 0
                    ready = False
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                if ready and current_time - last_capture_time >= 0.5:
                    features = self.extract_face_features(frame, (x, y, w, h))
                    if features is not None:
                        features_list.append(features)
                        face_positions.append(face_center)
                        count += 1
                        last_capture_time = current_time
                        stable_frames = 0
                        self.update_status(f"📸 {count}/{self.register_samples} авлаа!")
            
            # Draw progress
            self.draw_progress(frame, count, self.register_samples)
            
            # Display frame
            self.display_frame(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.video_capture.release()
        
        if len(features_list) >= 3:
            for features in features_list:
                self.known_face_features.append(features)
                self.known_face_names.append(self.register_name)
            
            self.update_status(f"✅ {self.register_name} амжилттай бүртгэгдлээ!")
            self.save_data()
            self.update_status_display()
        else:
            self.update_status(f"❌ Хангалттай зураг аваагүй!")
        
        self.stop_capture()
    
    def start_recognition(self):
        """Start face recognition"""
        if not self.known_face_features:
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
        
        self.update_status("\n🎥 Танилт эхэллээ...")
        
        threading.Thread(target=self.recognize_thread, daemon=True).start()
    
    def recognize_thread(self):
        """Recognition thread"""
        self.video_capture = cv2.VideoCapture(0)
        
        frame_count = 0
        last_results = {}
        
        while self.is_capturing:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 3 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5,
                    minSize=(60, 60), maxSize=(400, 400)
                )
                
                new_results = {}
                
                for face_id, (x, y, w, h) in enumerate(faces):
                    features = self.extract_face_features(frame, (x, y, w, h))
                    
                    if features is not None:
                        name, confidence = self.find_best_match(features)
                        new_results[face_id] = (x, y, w, h, name, confidence)
                
                last_results = new_results
            
            # Draw results
            for face_id, (x, y, w, h, name, confidence) in last_results.items():
                color = self.get_color(name, confidence)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label_y = y - 10 if y - 10 > 10 else y + h + 20
                cv2.rectangle(frame, (x, label_y - 25), (x+w, label_y), color, -1)
                
                text = f"{name} ({confidence:.0f}%)" if name != "Танигдаагүй" else name
                cv2.putText(frame, text, (x + 5, label_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            self.display_frame(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.video_capture.release()
        self.stop_capture()
    
    def stop_capture(self):
        """Stop video capture"""
        self.is_capturing = False
        if self.video_capture:
            self.video_capture.release()
        
        self.register_btn.config(state='normal')
        self.recognize_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        # Clear video label
        self.video_label.config(
            image='',
            text="Видео зогссон байна\n\n🎥 'Нүүр бүртгэх' эсвэл 'Танилт эхлүүлэх' дарна уу"
        )
    
    def display_frame(self, frame):
        """Display frame in GUI"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Resize to fit
        max_width = 800
        max_height = 600
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk, text='')
    
    def draw_progress(self, frame, current, total):
        """Draw progress bar"""
        bar_width = frame.shape[1] - 40
        bar_height = 30
        bar_x, bar_y = 20, frame.shape[0] - 50
        
        cv2.rectangle(frame, (bar_x-5, bar_y-5),
                     (bar_x + bar_width + 5, bar_y + bar_height + 5),
                     (50, 50, 50), -1)
        
        progress = int((current / total) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + progress, bar_y + bar_height),
                     (0, 255, 0), -1)
        
        text = f"{current}/{total}"
        cv2.putText(frame, text, (bar_x + bar_width//2 - 30, bar_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def is_new_angle(self, face_center, face_positions, min_diff=20):
        """Check if face position is new"""
        for prev_pos in face_positions:
            distance = np.sqrt((face_center[0] - prev_pos[0])**2 +
                             (face_center[1] - prev_pos[1])**2)
            if distance < min_diff:
                return False
        return True
    
    def extract_face_features(self, image, face_rect):
        """Extract face features"""
        try:
            x, y, w, h = face_rect
            face = image[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))
            
            if len(face_resized.shape) == 3:
                gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_resized
            
            gray_face = cv2.equalizeHist(gray_face)
            hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            return hist
        except:
            return None
    
    def find_best_match(self, features):
        """Find best matching face"""
        max_similarity = 0
        best_match_name = "Танигдаагүй"
        
        for idx, known_features in enumerate(self.known_face_features):
            similarity = np.dot(features, known_features) / (
                np.linalg.norm(features) * np.linalg.norm(known_features) + 1e-6
            )
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_name = self.known_face_names[idx]
        
        if max_similarity > self.threshold:
            return best_match_name, max_similarity * 100
        else:
            return "Танигдаагүй", 0
    
    def get_color(self, name, confidence):
        """Get color based on confidence"""
        if name != "Танигдаагүй":
            if confidence > 90:
                return (0, 255, 0)
            elif confidence > 85:
                return (0, 255, 255)
            else:
                return (0, 165, 255)
        return (0, 0, 255)
    
    def save_data(self):
        """Save face data"""
        if not self.known_face_features:
            messagebox.showwarning("Анхааруулга", "Хадгалах дата байхгүй!")
            return
        
        try:
            data = {
                'features': self.known_face_features,
                'names': self.known_face_names,
                'threshold': self.threshold,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.update_status("💾 Дата хадгалагдлаа!")
            messagebox.showinfo("Амжилт", "Дата хадгалагдлаа!")
        except Exception as e:
            messagebox.showerror("Алдаа", f"Хадгалахад алдаа: {e}")
    
    def load_data_silent(self):
        """Load data silently on startup"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_features = data['features']
                    self.known_face_names = data['names']
                    if 'threshold' in data:
                        self.threshold = data['threshold']
                        self.threshold_var.set(self.threshold)
            except:
                pass
    
    def load_data(self):
        """Load face data"""
        if not os.path.exists(self.data_file):
            messagebox.showwarning("Анхааруулга", "Дата файл олдсонгүй!")
            return
        
        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_features = data['features']
                self.known_face_names = data['names']
                if 'threshold' in data:
                    self.threshold = data['threshold']
                    self.threshold_var.set(self.threshold)
            
            self.update_status_display()
            messagebox.showinfo("Амжилт", "Дата ачаалагдлаа!")
        except Exception as e:
            messagebox.showerror("Алдаа", f"Ачаалахад алдаа: {e}")
    
    def show_people_list(self):
        """Show list of registered people"""
        if not self.known_face_names:
            messagebox.showinfo("Мэдээлэл", "Бүртгэлтэй хүн байхгүй")
            return
        
        name_counts = Counter(self.known_face_names)
        
        message = "📋 Бүртгэлтэй хүмүүс:\n\n"
        for name, count in sorted(name_counts.items()):
            message += f"👤 {name}: {count} зураг\n"
        
        messagebox.showinfo("Бүртгэлтэй хүмүүс", message)
    
    def delete_person(self):
        """Delete a person"""
        if not self.known_face_names:
            messagebox.showinfo("Мэдээлэл", "Бүртгэлтэй хүн байхгүй")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Хүн устгах")
        dialog.geometry("400x300")
        dialog.configure(bg='#2d2d44')
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(
            dialog,
            text="Устгах хүнийг сонгоно уу:",
            font=('Helvetica', 12),
            bg='#2d2d44',
            fg='#ffffff'
        ).pack(pady=20)
        
        name_counts = Counter(self.known_face_names)
        names = sorted(name_counts.keys())
        
        listbox = tk.Listbox(
            dialog,
            font=('Helvetica', 11),
            height=8
        )
        listbox.pack(fill='both', expand=True, padx=20, pady=10)
        
        for name in names:
            listbox.insert(tk.END, f"{name} ({name_counts[name]} зураг)")
        
        def delete_selected():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("Анхааруулга", "Хүн сонгоно уу!")
                return
            
            name = names[selection[0]]
            
            indices = [i for i, n in enumerate(self.known_face_names) if n == name]
            for idx in sorted(indices, reverse=True):
                del self.known_face_features[idx]
                del self.known_face_names[idx]
            
            self.update_status(f"🗑️ {name} устгагдлаа!")
            self.save_data()
            self.update_status_display()
            dialog.destroy()
        
        tk.Button(
            dialog,
            text="🗑️ Устгах",
            command=delete_selected,
            bg='#ff4444',
            fg='#ffffff',
            font=('Helvetica', 11, 'bold'),
            cursor='hand2'
        ).pack(pady=10)


def main():
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()


class FaceRecognitionSystem:
    def __init__(self, threshold=0.82, data_file="face_data.pkl"):
        self.known_face_features = []
        self.known_face_names = []
        self.data_file = data_file
        self.threshold = threshold

        # OpenCV нүүр олох classifier
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_eye.xml')

        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise Exception("❌ Haar Cascade файлууд ачаалагдсангүй!")

    def extract_face_features(self, image, face_rect):
        """Нүүрний онцлог шинж чанаруудыг гаргаж авах"""
        try:
            x, y, w, h = face_rect

            if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
                return None

            face = image[y:y+h, x:x+w]

            if face.size == 0:
                return None

            face_resized = cv2.resize(face, (100, 100))

            if len(face_resized.shape) == 3:
                gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_resized

            gray_face = cv2.equalizeHist(gray_face)

            hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            lbp_features = self.compute_lbp(gray_face)
            hog_features = self.compute_hog(gray_face)

            features = np.concatenate([hist, lbp_features, hog_features])

            return features
        except Exception as e:
            return None

    def compute_lbp(self, image):
        """Local Binary Pattern features"""
        height, width = image.shape
        radius = 1
        lbp = np.zeros((height-2*radius, width-2*radius), dtype=np.uint8)

        for i in range(radius, height-radius):
            for j in range(radius, width-radius):
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
                lbp[i-radius, j-radius] = code

        hist_lbp = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        hist_lbp = cv2.normalize(hist_lbp, hist_lbp).flatten()

        return hist_lbp

    def compute_hog(self, image):
        """HOG (Histogram of Oriented Gradients) features"""
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        bins = np.int32(angle / 40)
        bin_cells = []

        cell_size = 10
        for i in range(0, image.shape[0] - cell_size, cell_size):
            for j in range(0, image.shape[1] - cell_size, cell_size):
                cell_mag = mag[i:i+cell_size, j:j+cell_size]
                cell_angle = bins[i:i+cell_size, j:j+cell_size]

                hist = np.zeros(9)
                for k in range(9):
                    hist[k] = np.sum(cell_mag[cell_angle == k])

                bin_cells.extend(hist)

        hog_features = np.array(bin_cells)
        if np.linalg.norm(hog_features) > 0:
            hog_features = hog_features / np.linalg.norm(hog_features)

        return hog_features[:256]

    def auto_collect_face_data(self, name, num_samples=10, auto_save=True):
        """🤖 АВТОМАТ НҮҮР ТАНИУЛАХ - Phone Face ID шиг"""

        # Хэрэв энэ нэртэй хүн аль хэдийн байгаа бол сануулах
        if name in self.known_face_names:
            print(f"⚠️ '{name}' аль хэдийн бүртгэлтэй байна!")
            choice = input(
                "Юу хийх вэ?\n  1 - Шинэ зураг НЭМЭХ (сайжруулах)\n  2 - Өмнөхийг СОЛИХ (устгаад шинээр)\n  3 - Цуцлах\nСонголт: ").strip()

            if choice == '1':
                print(f"✅ {name}-д шинэ зургууд нэмэх горимд орлоо")
            elif choice == '2':
                indices = [i for i, n in enumerate(
                    self.known_face_names) if n == name]
                for idx in sorted(indices, reverse=True):
                    del self.known_face_features[idx]
                    del self.known_face_names[idx]
                print(f"🗑️ {name}-ын хуучин дата устгагдлаа, шинээр бүртгэнэ")
            elif choice == '3':
                print("🚫 Цуцлагдлаа")
                return False
            else:
                print("❌ Буруу сонголт, цуцлагдлаа")
                return False

        print(f"\n{'='*60}")
        print(f"📱 {name}-ын нүүрийг автоматаар бүртгэж байна...")
        print(f"🎯 {num_samples} өөр өнцгөөс зураг авна")
        print(f"💡 Толгойгоо аажуухан эргүүлээрэй")
        print(f"{'='*60}\n")

        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print("❌ Камер нээгдсэнгүй!")
            return False

        features_list = []
        count = 0
        last_capture_time = time.time()
        capture_interval = 0.5

        face_positions = []
        min_position_diff = 20

        stable_frames = 0
        min_stable_frames = 3

        print("🔍 Нүүрийг олж байна...")

        while count < num_samples:
            ret, frame = video_capture.read()
            if not ret:
                print("❌ Камераас frame уншиж чадсангүй!")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(100, 100), maxSize=(400, 400)
            )

            current_time = time.time()
            face_detected = False
            ready_to_capture = False

            for (x, y, w, h) in faces:
                face_detected = True

                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray, minNeighbors=8)

                has_eyes = len(eyes) >= 2

                face_center = (x + w//2, y + h//2)

                is_new_angle = True
                for prev_pos in face_positions:
                    distance = np.sqrt((face_center[0] - prev_pos[0])**2 +
                                       (face_center[1] - prev_pos[1])**2)
                    if distance < min_position_diff:
                        is_new_angle = False
                        break

                if has_eyes and is_new_angle:
                    color = (0, 255, 0)
                    stable_frames += 1
                    ready_to_capture = stable_frames >= min_stable_frames
                elif has_eyes:
                    color = (0, 255, 255)
                    stable_frames = 0
                else:
                    color = (0, 165, 255)
                    stable_frames = 0

                thickness = 3 if ready_to_capture else 2
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

                for (ex, ey, ew, eh) in eyes:
                    cv2.circle(frame, (x+ex+ew//2, y+ey+eh//2),
                               ew//2, (255, 0, 0), 2)

                if (ready_to_capture and is_new_angle and has_eyes and
                        current_time - last_capture_time >= capture_interval):

                    features = self.extract_face_features(frame, (x, y, w, h))

                    if features is not None:
                        features_list.append(features)
                        face_positions.append(face_center)
                        count += 1
                        last_capture_time = current_time
                        stable_frames = 0

                        cv2.circle(
                            frame, (frame.shape[1]//2, frame.shape[0]//2), 50, (0, 255, 0), 5)

                        print(f"📸 {count}/{num_samples} - ✓ Авлаа!")

            # Progress bar
            bar_width = frame.shape[1] - 40
            bar_height = 30
            bar_x, bar_y = 20, frame.shape[0] - 50

            cv2.rectangle(frame, (bar_x-5, bar_y-5),
                          (bar_x + bar_width + 5, bar_y + bar_height + 5),
                          (50, 50, 50), cv2.FILLED)

            progress = int((count / num_samples) * bar_width)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + progress, bar_y + bar_height),
                          (0, 255, 0), cv2.FILLED)

            progress_text = f"{count}/{num_samples}"
            cv2.putText(frame, progress_text,
                        (bar_x + bar_width//2 - 30, bar_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if face_detected:
                if ready_to_capture:
                    status = "📸 Авч байна..."
                    color = (0, 255, 0)
                elif not has_eyes:
                    status = "👀 Нүдийг харуулна уу"
                    color = (0, 165, 255)
                elif not is_new_angle:
                    status = "🔄 Толгойгоо эргүүлнэ үү"
                    color = (0, 255, 255)
                else:
                    status = "⏳ Бэлдэж байна..."
                    color = (255, 255, 0)
            else:
                status = "🔍 Нүүрийг олж байна..."
                color = (0, 0, 255)

            cv2.putText(frame, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            instruction = "Q - цуцлах | Автоматаар авна"
            cv2.putText(frame, instruction, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Auto Face ID', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🚫 Хэрэглэгч цуцалсан")
                break

        video_capture.release()
        cv2.destroyAllWindows()

        if len(features_list) >= 3:
            for features in features_list:
                self.known_face_features.append(features)
                self.known_face_names.append(name)

            print(f"\n{'='*60}")
            print(f"✅ {name} амжилттай бүртгэгдлээ!")
            print(f"📊 {len(features_list)} зураг хадгалсан")
            print(f"{'='*60}\n")

            if auto_save:
                self.save_data()

            return True
        else:
            print(
                f"\n❌ Хангалттай зураг аваагүй! ({len(features_list)}/{num_samples})")
            return False

    def collect_face_data_from_images(self, images_folder):
        """Зургийн фолдероос нүүрийг таниулах"""
        print(f"📸 {images_folder}-оос нүүрний дата цуглуулж байна...")

        if not os.path.exists(images_folder):
            print(f"❌ {images_folder} олдсонгүй!")
            return False

        image_files = [f for f in os.listdir(images_folder)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

        if not image_files:
            print("❌ Зураг олдсонгүй!")
            return False

        success_count = 0
        for filename in image_files:
            image_path = os.path.join(images_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"⚠️ {filename} уншиж чадсангүй")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(50, 50), maxSize=(500, 500)
            )

            if len(faces) > 0:
                face = max(faces, key=lambda rect: rect[2] * rect[3])
                features = self.extract_face_features(image, face)

                if features is not None:
                    name = os.path.splitext(
                        filename)[0].replace('_', ' ').title()
                    self.known_face_features.append(features)
                    self.known_face_names.append(name)
                    success_count += 1
                    print(f"✅ {name} таниулсан")
                else:
                    print(f"⚠️ {filename}-н features гаргаж чадсангүй")
            else:
                print(f"⚠️ {filename}-д нүүр олдсонгүй")

        print(f"\n📊 Нийт: {success_count}/{len(image_files)} нүүр таниулсан")

        if success_count > 0:
            save = input("\n💾 Одоо хадгалах уу? (y/n): ").strip().lower()
            if save == 'y' or save == 'yes':
                self.save_data()

        return success_count > 0

    def save_data(self):
        """Нүүрний датаг файлд хадгалах"""
        try:
            data = {
                'features': self.known_face_features,
                'names': self.known_face_names,
                'threshold': self.threshold,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)

            name_counts = Counter(self.known_face_names)

            print(f"💾 Дата хадгалагдлаа!")
            print(f"📁 Файл: {self.data_file}")
            print(f"👥 Хүмүүс: {len(name_counts)}")
            print(f"📊 Нийт зураг: {len(self.known_face_names)}")
        except Exception as e:
            print(f"❌ Хадгалахад алдаа гарлаа: {e}")

    def load_data(self):
        """Хадгалсан датаг ачаалах"""
        if not os.path.exists(self.data_file):
            print(f"❌ {self.data_file} файл олдсонгүй!")
            return False

        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_features = data['features']
                self.known_face_names = data['names']

                if 'threshold' in data:
                    self.threshold = data['threshold']

                if 'timestamp' in data:
                    print(f"📅 Хадгалсан огноо: {data['timestamp']}")

            name_counts = Counter(self.known_face_names)

            print(f"✅ Дата ачаалагдлаа!")
            print(
                f"👥 Хүмүүс ({len(name_counts)}): {', '.join(sorted(name_counts.keys()))}")
            print(f"📊 Нийт зураг: {len(self.known_face_names)}")
            return True
        except Exception as e:
            print(f"❌ Ачаалахад алдаа гарлаа: {e}")
            return False

    def compare_faces(self, features1, features2):
        """Хоёр нүүрийг харьцуулах"""
        cos_sim = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-6
        )

        euclidean_dist = np.linalg.norm(features1 - features2)
        euclidean_sim = 1 / (1 + euclidean_dist)

        similarity = 0.7 * cos_sim + 0.3 * euclidean_sim
        is_match = similarity > self.threshold

        return similarity, is_match

    def recognize_faces_video(self):
        """Видеогоор нүүр танилт хийх"""
        if not self.known_face_features:
            print("❌ Эхлээд дата ачаална уу эсвэл нүүр таниулна уу!")
            return

        print(f"\n🎥 Нүүр таних систем эхэллээ")
        print(f"👥 Бүртгэлтэй: {len(set(self.known_face_names))} хүн")
        print(f"📊 Нийт зураг: {len(self.known_face_names)}")
        print(f"🎯 Threshold: {self.threshold:.2f}")
        print("Q дарж гарна уу!\n")

        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print("❌ Камер нээгдсэнгүй!")
            return

        frame_skip = 3
        frame_count = 0

        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0

        last_results = {}

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            fps_frame_count += 1

            if fps_frame_count >= 30:
                elapsed = time.time() - fps_start_time
                fps = fps_frame_count / elapsed if elapsed > 0 else 0
                fps_start_time = time.time()
                fps_frame_count = 0

            if frame_count % frame_skip != 0:
                for face_id, (x, y, w, h, name, confidence) in last_results.items():
                    if name != "Танигдаагүй":
                        if confidence > 90:
                            color = (0, 255, 0)
                        elif confidence > 85:
                            color = (0, 255, 255)
                        else:
                            color = (0, 165, 255)
                    else:
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                    label_y = y - 10 if y - 10 > 10 else y + h + 20
                    cv2.rectangle(frame, (x, label_y - 25),
                                  (x+w, label_y), color, cv2.FILLED)

                    text = f"{name} ({confidence:.0f}%)" if name != "Танигдаагүй" else name
                    cv2.putText(frame, text, (x + 5, label_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow('Нүүр таних систем', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5,
                minSize=(60, 60), maxSize=(400, 400)
            )

            new_results = {}

            for face_id, (x, y, w, h) in enumerate(faces):
                features = self.extract_face_features(frame, (x, y, w, h))

                if features is None:
                    continue

                name = "Танигдаагүй"
                confidence = 0

                max_similarity = 0
                best_match_name = None

                for idx, known_features in enumerate(self.known_face_features):
                    similarity, _ = self.compare_faces(
                        known_features, features)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match_name = self.known_face_names[idx]

                if max_similarity > self.threshold and best_match_name:
                    name = best_match_name
                    confidence = max_similarity * 100

                new_results[face_id] = (x, y, w, h, name, confidence)

                if name != "Танигдаагүй":
                    if confidence > 90:
                        color = (0, 255, 0)
                    elif confidence > 85:
                        color = (0, 255, 255)
                    else:
                        color = (0, 165, 255)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                label_y = y - 10 if y - 10 > 10 else y + h + 20
                cv2.rectangle(frame, (x, label_y - 25),
                              (x+w, label_y), color, cv2.FILLED)

                text = f"{name} ({confidence:.0f}%)" if name != "Танигдаагүй" else name
                cv2.putText(frame, text, (x + 5, label_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            last_results = new_results

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"Нүүр: {len(faces)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Нүүр таних систем', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        print("\n👋 Систем хаагдлаа")

    def delete_person(self, name):
        """Хүний датаг устгах"""
        indices_to_remove = [i for i, n in enumerate(
            self.known_face_names) if n == name]

        if not indices_to_remove:
            print(f"❌ {name} олдсонгүй!")
            return False

        for idx in sorted(indices_to_remove, reverse=True):
            del self.known_face_features[idx]
            del self.known_face_names[idx]

        print(f"✅ {name} ({len(indices_to_remove)} зураг) устгагдлаа!")
        return True

    def list_people(self):
        """Бүртгэлтэй хүмүүсийг харуулах"""
        if not self.known_face_names:
            print("📋 Бүртгэлтэй хүн байхгүй")
            return

        name_counts = Counter(self.known_face_names)

        print(f"\n📋 Бүртгэлтэй хүмүүс ({len(name_counts)}):")
        print("=" * 50)
        for name, count in sorted(name_counts.items()):
            print(f"  👤 {name}: {count} зураг")
        print("=" * 50)


def main():
    # Энгийн файлын нэр ашиглах - одоогийн фолдерт хадгална
    system = FaceRecognitionSystem(
        threshold=0.72, data_file="C:/Users/troyz/OneDrive/Desktop/faceless/data/face_data.pkl")

    print("=" * 60)
    print("📱 AUTO FACE ID СИСТЕМ (Phone Face ID шиг)")
    print("=" * 60)

    # Өмнөх дата байвал ачаалах
    if os.path.exists(system.data_file):
        print("\n📂 Өмнөх дата олдлоо, ачаалж байна...")
        system.load_data()
    else:
        print("\n📝 Шинэ эхлэл - одоогоор хадгалсан дата байхгүй")

    while True:
        print("\n📋 ҮЙЛ АЖИЛЛАГАА:")
        print("  1 - 🤖 АВТОМАТ нүүр бүртгэх (Space дарах шаардлагагүй)")
        print("  2 - Зургийн фолдероос дата цуглуулах")
        print("  3 - Датаг хадгалах (гараар)")
        print("  4 - Датаг дахин ачаалах")
        print("  5 - Видеогоор танилт хийх")
        print("  6 - Бүртгэлтэй хүмүүсийг харах")
        print("  7 - Хүний датаг устгах")
        print(
            "  8 - Threshold тохируулах (одоо: {:.2f})".format(system.threshold))
        print("  9 - Бүх датаг устгах (reset)")
        print("  0 - Гарах")
        print("-" * 60)
        if system.known_face_names:
            print(
                f"💾 Одоогийн дата: {len(set(system.known_face_names))} хүн бүртгэлтэй")
        else:
            print("⚠️ Одоогоор бүртгэлтэй хүн байхгүй")
        print("-" * 60)

        choice = input("Сонголт: ").strip()

        if choice == '1':
            if system.known_face_names:
                print("\n📋 Одоо бүртгэлтэй хүмүүс:")
                unique_names = sorted(set(system.known_face_names))
                for i, person in enumerate(unique_names, 1):
                    count = system.known_face_names.count(person)
                    print(f"  {i}. {person} ({count} зураг)")
                print()

            name = input("Хүний нэр: ").strip()
            if name:
                num = input(
                    "Хэдэн өнцгөөс авах вэ? (5-15, default=10): ").strip()
                num = int(num) if num.isdigit() else 10
                system.auto_collect_face_data(name, num, auto_save=True)
            else:
                print("❌ Нэр оруулна уу!")

        elif choice == '2':
            if system.known_face_names:
                print("\n📋 Одоо бүртгэлтэй хүмүүс:")
                unique_names = sorted(set(system.known_face_names))
                for i, person in enumerate(unique_names, 1):
                    count = system.known_face_names.count(person)
                    print(f"  {i}. {person} ({count} зураг)")
                print()

            folder = input("Зургийн фолдерын зам: ").strip()
            if folder:
                system.collect_face_data_from_images(folder)
            else:
                print("❌ Фолдерын зам оруулна уу!")

        elif choice == '3':
            if system.known_face_features:
                system.save_data()
            else:
                print("❌ Хадгалах дата байхгүй!")

        elif choice == '4':
            system.load_data()

        elif choice == '5':
            system.recognize_faces_video()

        elif choice == '6':
            system.list_people()

        elif choice == '7':
            system.list_people()
            if system.known_face_names:
                name = input("\nУстгах хүний нэр: ").strip()
                if name and system.delete_person(name):
                    save = input(
                        "💾 Өөрчлөлтийг хадгалах уу? (y/n): ").strip().lower()
                    if save == 'y' or save == 'yes':
                        system.save_data()

        elif choice == '8':
            try:
                new_threshold = float(
                    input(f"Шинэ threshold (0.7-0.95, одоо={system.threshold:.2f}): "))
                if 0.7 <= new_threshold <= 0.95:
                    system.threshold = new_threshold
                    print(
                        f"✅ Threshold {new_threshold:.2f} болгож өөрчлөгдлөө")
                else:
                    print("❌ 0.7-0.95 хооронд утга оруулна уу!")
            except ValueError:
                print("❌ Буруу утга!")

        elif choice == '9':
            confirm = input(
                "⚠️ БҮХ ДАТАГ УСТГАХ уу? Буцаах боломжгүй! (yes гэж бичнэ үү): ").strip()
            if confirm.lower() == 'yes':
                system.known_face_features = []
                system.known_face_names = []
                if os.path.exists(system.data_file):
                    os.remove(system.data_file)
                    print("✅ Бүх дата устгагдлаа!")
                else:
                    print("✅ RAM дахь дата цэвэрлэгдлээ!")
            else:
                print("🚫 Цуцлагдлаа")

        elif choice == '0':
            # Гарахын өмнө хадгалаагүй өөрчлөлт байвал сануулах
            if system.known_face_features:
                needs_save = True
                if os.path.exists(system.data_file):
                    try:
                        with open(system.data_file, 'rb') as f:
                            saved_data = pickle.load(f)
                            if (len(saved_data['names']) == len(system.known_face_names) and
                                    saved_data['names'] == system.known_face_names):
                                needs_save = False
                    except:
                        needs_save = True

                if needs_save:
                    save_prompt = input(
                        "\n⚠️ Хадгалаагүй өөрчлөлт байна! Хадгалах уу? (y/n): ").strip().lower()
                    if save_prompt == 'y' or save_prompt == 'yes':
                        system.save_data()

            print("\n" + "=" * 60)
            print("👋 Баяртай! Нүүр таних систем хаагдаж байна...")
            print("=" * 60)
            break

        else:
            print("❌ Буруу сонголт! Дахин оролдоно уу.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Програм зогссон (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Алдаа гарлаа: {e}")
  