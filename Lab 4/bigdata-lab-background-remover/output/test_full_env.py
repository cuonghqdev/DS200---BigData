import sys
import os

# 1. Test Python Libraries
print("--- 1. Kiểm tra thư viện Python ---")
try:
    import cv2
    import mediapipe as mp
    import numpy as np
    print(f"✅ OpenCV version: {cv2.__version__}")
    print(f"✅ MediaPipe version: {mp.__version__}")
    print(f"✅ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"❌ THIẾU THƯ VIỆN: {e}")
    print("👉 Cậu chạy lệnh này nhé: pip install mediapipe opencv-python numpy")
    sys.exit(1)

# 2. Test Spark Session
print("\n--- 2. Kiểm tra Spark Session ---")
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("TestEnv") \
        .master("local[1]") \
        .getOrCreate()
    print(f"✅ Spark Version: {spark.version}")
    print("✅ Spark Session khởi tạo thành công!")
except Exception as e:
    print(f"❌ LỖI SPARK: {e}")
    sys.exit(1)

# 3. Test Model Path (Rất quan trọng)
print("\n--- 3. Kiểm tra file Model ---")
model_path = "models/selfie_segmenter.tflite"
if os.path.exists(model_path):
    print(f"✅ Đã tìm thấy model tại: {model_path}")
else:
    print(f"⚠️ CẢNH BÁO: Không thấy file '{model_path}'")
    print("👉 Hãy tạo thư mục 'models' và bỏ file .tflite vào đó nhé!")

print("\n🎉 CHÚC MỪNG! MÔI TRƯỜNG ĐÃ SẴN SÀNG ĐỂ CODE LAB.")