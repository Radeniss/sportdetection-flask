# Flask Mediapipe Pose Counter

Aplikasi Flask untuk menghitung repetisi push-up, squat, dan jumping jack dari video atau stream menggunakan MediaPipe Pose.

## Model MediaPipe
- Model: `mediapipe.solutions.pose` (varian BlazePose) dengan 33 landmark 2D + visibility.
- Parameter: `model_complexity=1`, `min_detection_confidence=0.5`, `min_tracking_confidence=0.5`.
- Input: frame BGR dikonversi ke RGB, bekerja sepenuhnya di CPU, cocok untuk real-time atau batch.
- Output: koordinat piksel dan skor visibilitas yang dipakai untuk menghitung sudut dan jarak sendi.

## Cara Kerja
1. Unggah video melalui API, file disimpan ke `static/uploads` lalu diproses di thread terpisah agar request cepat selesai.
2. Setiap frame dilewatkan ke MediaPipe Pose untuk mendapatkan 33 keypoint, kemudian diubah menjadi array `(x, y, visibility)`.
3. `RepetitionCounter` menghitung:
   - **Push-up**: memakai sudut siku (bahu-siku-pergelangan) dan validasi plank (bahu-pinggul-ankle) dengan ambang longgar 110/140 derajat.
   - **Squat**: sudut lutut kiri/kanan dan kedalaman pinggul dibanding lutut (ambang sekitar 90/160 derajat).
   - **Jumping Jack**: tangan di atas kepala + jarak antar mata kaki relatif ke lebar frame; dihitung saat transisi closed -> open.
4. Frame diberi overlay landmark dan teks jumlah repetisi, lalu disimpan sebagai video baru di `static/processed/processed_<nama_asli>`. Webhook menerima ringkasan hitungan.
5. Untuk streaming, `process_realtime_frame` memakai satu instance Pose yang dibagikan, sementara state counter per `stream_id` disimpan di memori dan bisa di-reset.

## Fitur API
- `POST /api/process` (`/process_video`): unggah `video`, sertakan `video_id` dan `webhook_url`; respon 202, pemrosesan berjalan di background.
- Webhook callback: mengirim `status`, `processed_filename`, dan hitungan `pushup`, `squat`, `jj` ke URL yang diberikan (atau `LARAVEL_WEBHOOK_URL`).
- `GET /api/download/<filename>` (`/download/<filename>`): unduh video hasil dengan overlay dan hitungan.
- `POST /api/realtime_frame`: kirim frame Base64 + `stream_id`, menerima frame teranotasi dan counter terkini.
- `POST /api/realtime_reset`: reset counter untuk stream tertentu.
- `GET /api/health`: pengecekan sederhana kesehatan service.

## Menjalankan Singkat
- Pasang dependensi: `pip install -r requirements.txt` (Gunakan Python 3.9+).
- Jalankan: `python app.py` (default `host=0.0.0.0`, `port=7000`).
- Upload video melalui endpoint atau gunakan alur realtime untuk streaming kamera.

## Catatan
- Video ditulis ulang dengan codec `avc1` dan fallback `mp4v` agar mudah diputar di browser.
- Deteksi mengandalkan visibilitas keypoint; jika bagian tubuh tidak terlihat lama, hitungan bisa melewatkan repetisi.
- Folder `static/uploads` dan `static/processed` dibuat otomatis jika belum ada.
