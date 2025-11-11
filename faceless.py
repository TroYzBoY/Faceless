import cv2
import face_recognition
import pickle
import os
import numpy as np
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.data_file = "face_data.pkl"
        
    # 1. ДАТАГ ЦУГЛУУЛАХ - Зураг эсвэл вебкамаас
    def collect_face_data_from_images(self, images_folder):
        """Зургийн фолдероос нүүрийг таниулах"""
        print("📸 Зургаас нүүрний дата цуглуулж байна...")
        
        for filename in os.listdir(images_folder):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(images_folder, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Нүүрийг олох
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    # Файлын нэрийг хүний нэр болгох (жишээ: "bataa.jpg" -> "bataa")
                    name = os.path.splitext(filename)[0]
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(name)
                    print(f"✅ {name} таниулсан")
                else:
                    print(f"❌ {filename}-д нүүр олдсонгүй")
    
    def collect_face_data_from_webcam(self, name, num_samples=5):
        """Вебкамаас нүүрний дата цуглуулах"""
        print(f"📹 {name}-ын нүүрийг {num_samples} удаа авах гэж байна...")
        print("Камер нээгдэх болно. 'Space' дарж зураг авна уу!")
        
        video_capture = cv2.VideoCapture(0)
        encodings = []
        count = 0
        
        while count < num_samples:
            ret, frame = video_capture.read()
            if not ret:
                break
                
            # Нүүр олох
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Нүүрийг зурах
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Мэдээлэл харуулах
            cv2.putText(frame, f"Авсан: {count}/{num_samples}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Space - зураг авах, Q - гарах", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Нүүр таниулах', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Space дарахад зураг авах
            if key == ord(' ') and face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings:
                    encodings.append(face_encodings[0])
                    count += 1
                    print(f"✅ Зураг {count} авлаа!")
            
            # Q дарахад гарах
            elif key == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        
        # Дундаж encoding авах
        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            self.known_face_encodings.append(avg_encoding)
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
            'encodings': self.known_face_encodings,
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
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print(f"✅ {len(self.known_face_names)} хүний дата ачаалагдлаа!")
            print(f"Хүмүүс: {', '.join(self.known_face_names)}")
            return True
        else:
            print(f"❌ {self.data_file} файл олдсонгүй!")
            return False
    
    # 4. ВИДЕОГООР ТАНИЛТ ХИЙХ
    def recognize_faces_video(self):
        """Видеогоор нүүр танилт хийх"""
        if not self.known_face_encodings:
            print("❌ Эхлээд дата ачаална уу!")
            return
        
        print("🎥 Нүүр таних систем эхэллээ...")
        print("Q дарж гарна уу!")
        
        video_capture = cv2.VideoCapture(0)
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Түргэн ажиллуулахын тулд зургийн хэмжээг багасгах
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Нүүр олох
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Таних
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                name = "Танигдаагүй"
                confidence = 0
                
                if True in matches:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = (1 - face_distances[best_match_index]) * 100
                
                # Зургийн хэмжээг буцаах
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Хүрээ зурах
                color = (0, 255, 0) if name != "Танигдаагүй" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                
                # Нэр бичих
                text = f"{name} ({confidence:.1f}%)" if name != "Танигдаагүй" else name
                cv2.putText(frame, text, (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Нүүр таних систем', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()

# ================ ХЭРЭГЛЭХ ЖИШЭЭ ================

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    
    print("🎯 НҮҮР ТАНИХ СИСТЕМ")
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
            print("❌ Буруу сонголт")