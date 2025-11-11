import cv2
import pickle
import os
import numpy as np

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_features = []
        self.known_face_names = []
        self.data_file = "face_data.pkl"
        
        # OpenCV нүүр олох classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def extract_face_features(self, image, face_rect):
        """Нүүрний онцлог шинж чанаруудыг гаргаж авах"""
        x, y, w, h = face_rect
        face = image[y:y+h, x:x+w]
        
        # Нүүрийг тогтмол хэмжээ болгох
        face_resized = cv2.resize(face, (100, 100))
        
        # Gray scale болгох
        gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Histogram ашиглах (энгийн feature vector)
        hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # LBP (Local Binary Pattern) хийх - нарийвчлал сайжруулах
        lbp_features = self.compute_lbp(gray_face)
        
        # Хоёрыг нэгтгэх
        features = np.concatenate([hist, lbp_features])
        
        return features
    
    def compute_lbp(self, image):
        """Local Binary Pattern features"""
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
        
        # LBP histogram
        hist_lbp = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        hist_lbp = cv2.normalize(hist_lbp, hist_lbp).flatten()
        
        return hist_lbp
    
    # 1. ДАТАГ ЦУГЛУУЛАХ - Зураг эсвэл вебкамаас
    def collect_face_data_from_images(self, images_folder):
        """Зургийн фолдероос нүүрийг таниулах"""
        print("📸 Зургаас нүүрний дата цуглуулж байна...")
        
        for filename in os.listdir(images_folder):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(images_folder, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"❌ {filename} уншиж чадсангүй")
                    continue
                
                # Нүүр олох
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) > 0:
                    # Хамгийн том нүүрийг авах
                    face = max(faces, key=lambda rect: rect[2] * rect[3])
                    features = self.extract_face_features(image, face)
                    
                    # Файлын нэрийг хүний нэр болгох
                    name = os.path.splitext(filename)[0]
                    self.known_face_features.append(features)
                    self.known_face_names.append(name)
                    print(f"✅ {name} таниулсан")
                else:
                    print(f"❌ {filename}-д нүүр олдсонгүй")
    
    def collect_face_data_from_webcam(self, name, num_samples=5):
        """Вебкамаас нүүрний дата цуглуулах"""
        print(f"📹 {name}-ын нүүрийг {num_samples} удаа авах гэж байна...")
        print("Камер нээгдэх болно. 'Space' дарж зураг авна уу!")
        
        video_capture = cv2.VideoCapture(0)
        features_list = []
        count = 0
        
        while count < num_samples:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Нүүр олох
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Нүүрүүдийг зурах
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Нүдийг олох (нарийвчлал сайжруулах)
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
            
            # Мэдээлэл харуулах
            cv2.putText(frame, f"Авсан: {count}/{num_samples}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Space - зураг авах, Q - гарах", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if len(faces) > 0:
                cv2.putText(frame, "Нүүр олдлоо! Space дарна уу", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Нүүр таниулах', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Space дарахад зураг авах
            if key == ord(' ') and len(faces) > 0:
                # Хамгийн том нүүрийг авах
                face = max(faces, key=lambda rect: rect[2] * rect[3])
                features = self.extract_face_features(frame, face)
                features_list.append(features)
                count += 1
                print(f"✅ Зураг {count} авлаа!")
            
            # Q дарахад гарах
            elif key == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        
        # Дундаж features авах
        if features_list:
            avg_features = np.mean(features_list, axis=0)
            self.known_face_features.append(avg_features)
            self.known_face_names.append(name)
            print(f"✅ {name} амжилттай таниулсан!")
            return True
        else:
            print(f"❌ Нүүр олдсонгүй!")
            return False
    
    # 2. ДАТАГ ХАДГАЛАХ
    def save_data(self):
        """Нүүрний датаг файлд хадгалах"""
        data = {
            'features': self.known_face_features,
            'names': self.known_face_names
        }
        with open(self.data_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Дата {self.data_file} файлд хадгалагдлаа!")
    
    # 3. ДАТАГ УНШИЖ АЧААЛАХ
    def load_data(self):
        """Хадгалсан датаг ачаалах"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_features = data['features']
                self.known_face_names = data['names']
            print(f"✅ {len(self.known_face_names)} хүний дата ачаалагдлаа!")
            print(f"Хүмүүс: {', '.join(self.known_face_names)}")
            return True
        else:
            print(f"❌ {self.data_file} файл олдсонгүй!")
            return False
    
    def compare_faces(self, features1, features2):
        """Хоёр нүүрийг харьцуулах"""
        # Cosine similarity ашиглах
        similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        # Similarity их байх тусам нүүр ижил (0-1 хооронд)
        return similarity, similarity > 0.85  # Threshold: 0.85
    
    # 4. ВИДЕОГООР ТАНИЛТ ХИЙХ
    def recognize_faces_video(self):
        """Видеогоор нүүр танилт хийх"""
        if not self.known_face_features:
            print("❌ Эхлээд дата ачаална уу!")
            return
        
        print("🎥 Нүүр таних систем эхэллээ...")
        print("Q дарж гарна уу!")
        
        video_capture = cv2.VideoCapture(0)
        frame_count = 0
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Хурдыг сайжруулахын тулд 3 frame тутамд танилт хийх
            if frame_count % 3 != 0:
                cv2.imshow('Нүүр таних систем', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # Нүүрний features гаргах
                features = self.extract_face_features(frame, (x, y, w, h))
                
                name = "Танигдаагүй"
                confidence = 0
                
                # Бүх таниулсан нүүртэй харьцуулах
                max_similarity = 0
                best_match_idx = -1
                
                for idx, known_features in enumerate(self.known_face_features):
                    similarity, is_match = self.compare_faces(known_features, features)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match_idx = idx
                
                # Threshold шалгах
                if max_similarity > 0.85:  # Энэ утгыг тохируулж болно
                    name = self.known_face_names[best_match_idx]
                    confidence = max_similarity * 100
                
                # Хүрээ зурах
                color = (0, 255, 0) if name != "Танигдаагүй" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Нэрний өнгөтэй дэвсгэр зурах
                cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
                
                # Нэр бичих
                text = f"{name} ({confidence:.1f}%)" if name != "Танигдаагүй" else name
                cv2.putText(frame, text, (x + 6, y - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Нүүр таних систем', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()

# ================ ХЭРЭГЛЭХ ЖИШЭЭ ================

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    
    print("🎯 НҮҮР ТАНИХ СИСТЕМ (OpenCV Haar Cascade)")
    print("=" * 50)
    print("1 - Вебкамаас дата цуглуулах")
    print("2 - Зургаас дата цуглуулах")
    print("3 - Датаг хадгалах")
    print("4 - Датаг ачаалах")
    print("5 - Видеогоор танилт хийх")
    print("0 - Гарах")
    print("=" * 50)
    
    while True:
        choice = input("\nСонголт оруулна уу: ")
        
        if choice == '1':
            name = input("Хүний нэр: ")
            num = int(input("Хэдэн зураг авах вэ? (5-10 санал болгох): "))
            system.collect_face_data_from_webcam(name, num)
            
        elif choice == '2':
            folder = input("Зургийн фолдер: ")
            if os.path.exists(folder):
                system.collect_face_data_from_images(folder)
            else:
                print("❌ Фолдер олдсонгүй!")
                
        elif choice == '3':
            system.save_data()
            
        elif choice == '4':
            system.load_data()
            
        elif choice == '5':
            system.recognize_faces_video()
            
        elif choice == '0':
            print("👋 Баяртай!")
            break
        else:
            print("❌ Буруу сонголт!")