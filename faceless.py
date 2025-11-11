import cv2
import pickle
import os
import numpy as np
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self, threshold=0.85):
        self.known_face_features = []
        self.known_face_names = []
        self.data_file = "face_data.pkl"
        self.threshold = threshold  # Танилтын босго утга
        
        # OpenCV нүүр олох classifier
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_eye.xml')
        
        # Classifier амжилттай ачаалагдсан эсэхийг шалгах
        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise Exception("❌ Haar Cascade файлууд ачаалагдсангүй!")
        
    def extract_face_features(self, image, face_rect):
        """Нүүрний онцлог шинж чанаруудыг гаргаж авах"""
        try:
            x, y, w, h = face_rect
            
            # Хүрээний шалгалт
            if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
                return None
            
            face = image[y:y+h, x:x+w]
            
            if face.size == 0:
                return None
            
            # Нүүрийг тогтмол хэмжээ болгох
            face_resized = cv2.resize(face, (100, 100))
            
            # Gray scale болгох
            if len(face_resized.shape) == 3:
                gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_resized
            
            # Гэрэлтүүлгийн тогтворжуулалт
            gray_face = cv2.equalizeHist(gray_face)
            
            # Histogram features
            hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # LBP features
            lbp_features = self.compute_lbp(gray_face)
            
            # HOG features (илүү сайн танилт)
            hog_features = self.compute_hog(gray_face)
            
            # Бүх features-ийг нэгтгэх
            features = np.concatenate([hist, lbp_features, hog_features])
            
            return features
        except Exception as e:
            print(f"⚠️ Features гаргахад алдаа: {e}")
            return None
    
    def compute_lbp(self, image):
        """Local Binary Pattern features - сайжруулсан хувилбар"""
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
        
        # LBP histogram - uniform patterns ашиглах
        hist_lbp = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        hist_lbp = cv2.normalize(hist_lbp, hist_lbp).flatten()
        
        return hist_lbp
    
    def compute_hog(self, image):
        """HOG (Histogram of Oriented Gradients) features"""
        # Gradient тооцоолох
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
        
        # Magnitude болон angle
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        # Histogram (9 bins)
        bins = np.int32(angle / 40)  # 0-360 -> 0-8
        bin_cells = []
        
        # 10x10 cell тус бүрээс histogram авах
        cell_size = 10
        for i in range(0, image.shape[0] - cell_size, cell_size):
            for j in range(0, image.shape[1] - cell_size, cell_size):
                cell_mag = mag[i:i+cell_size, j:j+cell_size]
                cell_angle = bins[i:i+cell_size, j:j+cell_size]
                
                hist = np.zeros(9)
                for k in range(9):
                    hist[k] = np.sum(cell_mag[cell_angle == k])
                
                bin_cells.extend(hist)
        
        # Normalize
        hog_features = np.array(bin_cells)
        if np.linalg.norm(hog_features) > 0:
            hog_features = hog_features / np.linalg.norm(hog_features)
        
        return hog_features[:256]  # Хэмжээг хязгаарлах
    
    def collect_face_data_from_images(self, images_folder):
        """Зургийн фолдероос нүүрийг таниулах - сайжруулсан"""
        print(f"📸 {images_folder}-оос нүүрний дата цуглуулж байна...")
        
        if not os.path.exists(images_folder):
            print(f"❌ {images_folder} олдсонгүй!")
            return
        
        image_files = [f for f in os.listdir(images_folder) 
                      if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        
        if not image_files:
            print("❌ Зураг олдсонгүй!")
            return
        
        success_count = 0
        for filename in image_files:
            image_path = os.path.join(images_folder, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"⚠️ {filename} уншиж чадсангүй")
                continue
            
            # Нүүр олох
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, 
                minSize=(50, 50), maxSize=(500, 500)
            )
            
            if len(faces) > 0:
                # Хамгийн том нүүрийг авах
                face = max(faces, key=lambda rect: rect[2] * rect[3])
                features = self.extract_face_features(image, face)
                
                if features is not None:
                    # Файлын нэрийг хүний нэр болгох (extension-г авах)
                    name = os.path.splitext(filename)[0].replace('_', ' ').title()
                    self.known_face_features.append(features)
                    self.known_face_names.append(name)
                    success_count += 1
                    print(f"✅ {name} таниулсан")
                else:
                    print(f"⚠️ {filename}-н features гаргаж чадсангүй")
            else:
                print(f"⚠️ {filename}-д нүүр олдсонгүй")
        
        print(f"\n📊 Нийт: {success_count}/{len(image_files)} нүүр таниулсан")
    
    def collect_face_data_from_webcam(self, name, num_samples=10):
        """Вебкамаас нүүрний дата цуглуулах - сайжруулсан"""
        print(f"📹 {name}-ын нүүрийг {num_samples} удаа авах гэж байна...")
        print("💡 Өөр өөр өнцөг, гэрэлтүүлэгээр зураг авбал сайн!")
        
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("❌ Камер нээгдсэнгүй!")
            return False
        
        features_list = []
        count = 0
        
        while count < num_samples:
            ret, frame = video_capture.read()
            if not ret:
                print("❌ Камераас frame уншиж чадсангүй!")
                break
            
            # Нүүр олох
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            
            # Нүүрүүдийг зурах
            face_detected = False
            for (x, y, w, h) in faces:
                face_detected = True
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Нүдийг олох
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, minNeighbors=8)
                for (ex, ey, ew, eh) in eyes:
                    cv2.circle(frame, (x+ex+ew//2, y+ey+eh//2), ew//2, (255, 0, 0), 2)
            
            # Прогресс мэдээлэл
            progress_text = f"Авсан: {count}/{num_samples}"
            cv2.putText(frame, progress_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            instruction = "SPACE - зураг авах | Q - гарах"
            cv2.putText(frame, instruction, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if face_detected:
                status = "✓ Нүүр олдлоо! SPACE дарна уу"
                color = (0, 255, 0)
            else:
                status = "✗ Нүүр олохыг оролдож байна..."
                color = (0, 0, 255)
            
            cv2.putText(frame, status, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow('Нүүр таниулах', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Space дарахад зураг авах
            if key == ord(' ') and len(faces) > 0:
                face = max(faces, key=lambda rect: rect[2] * rect[3])
                features = self.extract_face_features(frame, face)
                
                if features is not None:
                    features_list.append(features)
                    count += 1
                    print(f"✅ Зураг {count}/{num_samples} авлаа!")
                else:
                    print("⚠️ Features гаргаж чадсангүй, дахин оролдоно уу")
            
            # Q дарахад гарах
            elif key == ord('q'):
                print("🚫 Цуцлагдлаа")
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        
        # Дундаж features авах
        if len(features_list) >= 3:  # Хамгийн багадаа 3 зураг
            avg_features = np.mean(features_list, axis=0)
            self.known_face_features.append(avg_features)
            self.known_face_names.append(name)
            print(f"✅ {name} амжилттай таниулсан! ({len(features_list)} зураг)")
            return True
        else:
            print(f"❌ Хангалттай зураг аваагүй! ({len(features_list)}/{num_samples})")
            return False
    
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
            print(f"💾 {len(self.known_face_names)} хүний дата хадгалагдлаа!")
            print(f"📁 Файл: {self.data_file}")
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
            
            print(f"✅ {len(self.known_face_names)} хүний дата ачаалагдлаа!")
            print(f"👤 Хүмүүс: {', '.join(set(self.known_face_names))}")
            return True
        except Exception as e:
            print(f"❌ Ачаалахад алдаа гарлаа: {e}")
            return False
    
    def compare_faces(self, features1, features2):
        """Хоёр нүүрийг харьцуулах - олон аргаар"""
        # 1. Cosine similarity
        cos_sim = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-6
        )
        
        # 2. Euclidean distance (normalized)
        euclidean_dist = np.linalg.norm(features1 - features2)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # Хоёрыг хослуулах (weighted average)
        similarity = 0.7 * cos_sim + 0.3 * euclidean_sim
        
        is_match = similarity > self.threshold
        
        return similarity, is_match
    
    def recognize_faces_video(self):
        """Видеогоор нүүр танилт хийх - сайжруулсан"""
        if not self.known_face_features:
            print("❌ Эхлээд дата ачаална уу эсвэл нүүр таниулна уу!")
            return
        
        print(f"🎥 Нүүр таних систем эхэллээ ({len(self.known_face_names)} хүн)")
        print("Q дарж гарна уу!")
        
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("❌ Камер нээгдсэнгүй!")
            return
        
        frame_count = 0
        fps_start_time = datetime.now()
        fps = 0
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            
            # FPS тооцоолох
            if frame_count % 30 == 0:
                fps_end_time = datetime.now()
                time_diff = (fps_end_time - fps_start_time).total_seconds()
                fps = 30 / time_diff if time_diff > 0 else 0
                fps_start_time = fps_end_time
            
            # Хурдыг сайжруулахын тулд 2 frame тутамд танилт хийх
            if frame_count % 2 != 0:
                cv2.imshow('Нүүр таних систем', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, 
                minSize=(50, 50), maxSize=(500, 500)
            )
            
            for (x, y, w, h) in faces:
                # Нүүрний features гаргах
                features = self.extract_face_features(frame, (x, y, w, h))
                
                if features is None:
                    continue
                
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
                if max_similarity > self.threshold:
                    name = self.known_face_names[best_match_idx]
                    confidence = max_similarity * 100
                
                # Хүрээ зурах - өнгө нь итгэлийн түвшинээс хамаарна
                if name != "Танигдаагүй":
                    if confidence > 90:
                        color = (0, 255, 0)  # Ногоон - маш сайн
                    elif confidence > 85:
                        color = (0, 255, 255)  # Шар - дунд зэрэг
                    else:
                        color = (0, 165, 255)  # Улбар шар
                else:
                    color = (0, 0, 255)  # Улаан - танигдаагүй
                
                # Хүрээ
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Нэрний дэвсгэр
                label_y = y - 10 if y - 10 > 10 else y + h + 20
                cv2.rectangle(frame, (x, label_y - 25), (x+w, label_y), color, cv2.FILLED)
                
                # Нэр бичих
                text = f"{name} ({confidence:.0f}%)" if name != "Танигдаагүй" else name
                cv2.putText(frame, text, (x + 5, label_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # FPS мэдээлэл
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Танигдсан нүүрийн тоо
            cv2.putText(frame, f"Нүүр: {len(faces)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Нүүр таних систем', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        print("👋 Систем хаагдлаа")
    
    def delete_person(self, name):
        """Хүний датаг устгах"""
        indices_to_remove = [i for i, n in enumerate(self.known_face_names) if n == name]
        
        if not indices_to_remove:
            print(f"❌ {name} олдсонгүй!")
            return False
        
        # Урвуу дарааллаар устгах (index өөрчлөгдөхгүйн тулд)
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
        
        from collections import Counter
        name_counts = Counter(self.known_face_names)
        
        print(f"\n📋 Бүртгэлтэй хүмүүс ({len(name_counts)}):")
        print("=" * 50)
        for name, count in sorted(name_counts.items()):
            print(f"  👤 {name}: {count} зураг")
        print("=" * 50)

# ================ ХЭРЭГЛЭХ ЖИШЭЭ ================

def main():
    system = FaceRecognitionSystem(threshold=0.85)
    
    print("=" * 60)
    print("🎯 НҮҮР ТАНИХ СИСТЕМ (OpenCV + Haar Cascade + HOG + LBP)")
    print("=" * 60)
    
    while True:
        print("\n📋 ҮЙЛ АЖИЛЛАГАА:")
        print("  1 - Вебкамаас дата цуглуулах")
        print("  2 - Зургийн фолдероос дата цуглуулах")
        print("  3 - Датаг хадгалах")
        print("  4 - Датаг ачаалах")
        print("  5 - Видеогоор танилт хийх")
        print("  6 - Бүртгэлтэй хүмүүсийг харах")
        print("  7 - Хүний датаг устгах")
        print("  8 - Threshold тохируулах (одоо: {:.2f})".format(system.threshold))
        print("  0 - Гарах")
        print("-" * 60)
        
        choice = input("Сонголт: ").strip()
        
        if choice == '1':
            name = input("Хүний нэр: ").strip()
            if name:
                num = input("Хэдэн зураг авах вэ? (5-15, default=10): ").strip()
                num = int(num) if num.isdigit() else 10
                system.collect_face_data_from_webcam(name, num)
            else:
                print("❌ Нэр оруулна уу!")
                
        elif choice == '2':
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
            name = input("\nУстгах хүний нэр: ").strip()
            if name:
                system.delete_person(name)
            
        elif choice == '8':
            try:
                new_threshold = float(input(f"Шинэ threshold (0.7-0.95, одоо={system.threshold:.2f}): "))
                if 0.7 <= new_threshold <= 0.95:
                    system.threshold = new_threshold
                    print(f"✅ Threshold {new_threshold:.2f} болгож өөрчлөгдлөө")
                else:
                    print("❌ 0.7-0.95 хооронд утга оруулна уу!")
            except ValueError:
                print("❌ Буруу утга!")
                
        elif choice == '0':
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
        1