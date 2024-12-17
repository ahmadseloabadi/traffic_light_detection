import cv2
import numpy as np
import time
from tkinter import Tk, filedialog

class TrafficLightDetector:
    def __init__(self, log_file="traffic_light_log.txt", low_light_correction=True):
        """
        Inisialisasi detektor lampu lalu lintas.
        :param log_file: Nama file log untuk mencatat hasil deteksi.
        :param low_light_correction: Aktifkan koreksi pencahayaan rendah.
        """
        self.log_file = log_file
        self.low_light_correction = low_light_correction

    def preprocess_frame(self, frame):
        """
        Melakukan preprocessing pada frame, termasuk koreksi low-light jika diaktifkan.
        :param frame: Frame video dalam format BGR.
        :return: Frame dalam ruang warna HSV.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Koreksi pencahayaan rendah (low-light) menggunakan histogram equalization
        if self.low_light_correction:
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        return hsv

    def detect_colors(self, hsv_frame):
        """
        Mendeteksi warna lampu lalu lintas (merah, kuning, hijau) dalam ruang warna HSV.
        :param hsv_frame: Frame dalam format HSV.
        :return: Mask untuk masing-masing warna (red, yellow, green).
        """
        # Rentang HSV untuk deteksi warna
        red_lower1 = np.array([0, 120, 70])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 120, 70])
        red_upper2 = np.array([180, 255, 255])

        yellow_lower = np.array([15, 100, 100])
        yellow_upper = np.array([35, 255, 255])

        green_lower = np.array([40, 100, 100])
        green_upper = np.array([90, 255, 255])

        # Mask warna
        red_mask = cv2.add(cv2.inRange(hsv_frame, red_lower1, red_upper1),
                           cv2.inRange(hsv_frame, red_lower2, red_upper2))
        yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

        return red_mask, yellow_mask, green_mask

    def find_and_draw_contours(self, frame, masks, colors):
        """
        Menemukan kontur dan menggambar bounding box pada frame untuk lampu yang terdeteksi.
        :param frame: Frame video dalam format BGR.
        :param masks: Daftar mask untuk masing-masing warna.
        :param colors: Daftar warna dalam format (label, BGR).
        :return: Frame dengan bounding box dan daftar warna yang terdeteksi.
        """
        detected_colors = []
        for mask, (label, bgr_color) in zip(masks, colors):
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if  area > 250 :  # Filter area kecil
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)
                    detected_colors.append(label)
        return frame, detected_colors

    def write_log(self, detected_colors, timestamp):
        """
        Menulis log deteksi ke file log.
        :param detected_colors: Daftar warna yang terdeteksi.
        :param timestamp: Timestamp deteksi.
        """
        with open(self.log_file, "a") as log_file:
            for color in detected_colors:
                log_file.write(f"{timestamp} - Detected: {color}\n")

   
    def select_video_file(self):
        """
        memilih input file video
        """
        root = Tk()
        root.withdraw()  # Sembunyikan jendela utama tkinter
        video_path = filedialog.askopenfilename(title="Pilih File Video",
                                                filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
        return video_path

    def run(self):
        """
        Menjalankan detektor pada video input.
        """
        video_path = self.select_video_file()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Preprocess frame dan deteksi warna
            hsv_frame = self.preprocess_frame(frame)
            masks = self.detect_colors(hsv_frame)

            # Warna dan label untuk bounding box
            colors = [("Red", (0, 0, 255)), ("Yellow", (0, 255, 255)), ("Green", (0, 255, 0))]

            # Gambar bounding box dan dapatkan warna yang terdeteksi
            frame, detected_colors = self.find_and_draw_contours(frame, masks, colors)

            # Tulis log jika ada deteksi
            if detected_colors:
                self.write_log(detected_colors, timestamp)

            # Tampilkan frame
            cv2.imshow("Traffic Light Detection", frame)

            # Tekan 'q' untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Jalankan detektor
if __name__ == "__main__":
    # Masukkan path ke video Anda
    
    detector = TrafficLightDetector()
    detector.run()
