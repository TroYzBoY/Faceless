import cv2
import pickle
import os
import numpy as np
from datetime import datetime
import time


class FaceRecognitionSystem:
    def __init__(self, threshold=0.82, data_file="face_data.pkl"):
        self.known_face_features = []
        self.known_face_names = []
        self.data_file = data_file  # Одоо өөрчлөх боломжтой
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
        """
        🤖 АВТОМАТ НҮҮР ТАНИУЛАХ - Phone Face ID шиг
        Автоматаар нүүрийг олж, зураг аваад, хадгална
        """
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
        min_stable_frames = 3  # Багасгасан - хурдан болгох

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

                if (ready_to_capture and
                    is_new_angle and
                    has_eyes and
                        current_time - last_capture_time >= capture_interval):

                    features = self.extract_face_features(frame, (x, y, w, h))

                    if features is not None:
                        features_list.append(features)
                        face_positions.append(face_center)
                        count += 1
                        last_capture_time = current_time
                        stable_frames = 0

                        cv2.circle(frame, (frame.shape[1]//2, frame.shape[0]//2),
                                   50, (0, 255, 0), 5)

                        print(f"📸 {count}/{num_samples} - ✓ Авлаа!")

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
            # ХУУЧИн КОД: зөвхөн дундаж feature хадгалж байсан
            # avg_features = np.mean(features_list, axis=0)
            # self.known_face_features.append(avg_features)
            # self.known_face_names.append(name)

            # ШИНЭ КОД: Бүх features-ийг хадгалах (илүү сайн танилт)
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

            from collections import Counter
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

            from collections import Counter
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
        """Видеогоор нүүр танилт хийх - ОНОВЧЛОГДСОН"""
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

        # Хурд сайжруулах параметрүүд
        frame_skip = 3  # 3 frame тутамд танилт хийх
        frame_count = 0

        # FPS тооцоолох
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0

        # Сүүлийн танилтын үр дүн хадгалах
        last_results = {}

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            fps_frame_count += 1

            # FPS тооцоолох
            if fps_frame_count >= 30:
                elapsed = time.time() - fps_start_time
                fps = fps_frame_count / elapsed if elapsed > 0 else 0
                fps_start_time = time.time()
                fps_frame_count = 0

            # Frame skip - хурд сайжруулах
            if frame_count % frame_skip != 0:
                # Сүүлийн үр дүнг харуулах
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

            # Нүүр олох
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5,
                minSize=(60, 60), maxSize=(400, 400)
            )

            # Шинэ үр дүн хадгалах
            new_results = {}

            for face_id, (x, y, w, h) in enumerate(faces):
                features = self.extract_face_features(frame, (x, y, w, h))

                if features is None:
                    continue

                name = "Танигдаагүй"
                confidence = 0

                max_similarity = 0
                best_match_name = None

                # Бүх хадгалсан features-тай харьцуулах
                for idx, known_features in enumerate(self.known_face_features):
                    similarity, _ = self.compare_faces(
                        known_features, features)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match_name = self.known_face_names[idx]

                if max_similarity > self.threshold and best_match_name:
                    name = best_match_name
                    confidence = max_similarity * 100

                # Үр дүн хадгалах
                new_results[face_id] = (x, y, w, h, name, confidence)

                # Хүрээ зурах
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

            # Сүүлийн үр дүнг шинэчлэх
            last_results = new_results

            # Мэдээлэл харуулах
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

        from collections import Counter
        name_counts = Counter(self.known_face_names)

        print(f"\n📋 Бүртгэлтэй хүмүүс ({len(name_counts)}):")
        print("=" * 50)
        for name, count in sorted(name_counts.items()):
            print(f"  👤 {name}: {count} зураг")
        print("=" * 50)


def main():
    # ӨӨРИЙН ЗАМАА ЭНДЕ ОРУУЛНА УУ!
    # Жишээ: data_file="C:/Users/YourName/Desktop/data/face_data.pkl"
    system = FaceRecognitionSystem(threshold=0.82, data_file="C:/Users/troyz/OneDrive/Desktop/faceless/data/face_data.pkl")
    print("=" * 60)
    print("📱 AUTO FACE ID СИСТЕМ (Phone Face ID шиг)")
    print("=" * 60)
    print(f"📁 Дата файл: {system.data_file}\n")

    while True:
        print("\n📋 ҮЙЛ АЖИЛЛАГАА:")
        print("  1 - 🤖 АВТОМАТ нүүр бүртгэх (Space дарах шаардлагагүй)")
        print("  2 - Зургийн фолдероос дата цуглуулах")
        print("  3 - Датаг хадгалах")
        print("  4 - Датаг ачаалах")
        print("  5 - Видеогоор танилт хийх")
        print("  6 - Бүртгэлтэй хүмүүсийг харах")
        print("  7 - Хүний датаг устгах")
        print(
            "  8 - Threshold тохируулах (одоо: {:.2f})".format(system.threshold))
        print("  0 - Гарах")
        print("-" * 60)

        choice = input("Сонголт: ").strip()

        if choice == '1':
            name = input("Хүний нэр: ").strip()
            if name:
                num = input(
                    "Хэдэн өнцгөөс авах вэ? (5-15, default=10): ").strip()
                num = int(num) if num.isdigit() else 10
                system.auto_collect_face_data(name, num, auto_save=True)
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
                if system.delete_person(name):
                    # Устгасан бол автоматаар хадгалах
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