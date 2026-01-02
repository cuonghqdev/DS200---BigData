import os
import sys

# 1. CẤU HÌNG
os.environ['HADOOP_HOME'] = "C:\\hadoop"
os.environ['hadoop.home.dir'] = "C:\\hadoop"
if "C:\\hadoop\\bin" not in os.environ['PATH']:
    os.environ['PATH'] += ";" + "C:\\hadoop\\bin"

# Ép Spark dùng đúng Python của môi trường Conda
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import base64
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType

# Cấu hình đường dẫn
MODEL_PATH = "models/selfie_segmenter.tflite"
OUTPUT_DIR = "output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 2. LOGIC XÓA PHÔNG 
def process_frame_logic(base64_img_str, frame_id):
    try:
        # A. Decode ảnh
        img_bytes = base64.b64decode(base64_img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None: return "ERR: Frame None"
        
        # B. Setup MediaPipe
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
        segmenter = vision.ImageSegmenter.create_from_options(options)

        # C. Chạy AI lấy Mask
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask.numpy_view()


        # 1. Lấy kích thước ảnh gốc (Height, Width)
        h, w = frame.shape[:2]
        
        # 2. Resize Mask về đúng y chang kích thước ảnh gốc
        # Lưu ý: cv2.resize nhận (Width, Height)
        mask_resized = cv2.resize(category_mask, (w, h))

        # 3. Biến đổi Mask thành 3 kênh màu (giống hệt ảnh gốc)
        condition_bool = mask_resized > 0.1
        condition_float = condition_bool.astype(np.float32) # Chuyển về 0.0 và 1.0
        
        # Gộp 3 cái mask 1 kênh thành 1 cái mask 3 kênh
        mask_3d = cv2.merge([condition_float, condition_float, condition_float])
        
        # 4. Kiểm tra lần cuối 
        if mask_3d.shape != frame.shape:
             return f"LOI: Mask={mask_3d.shape} != Frame={frame.shape}"

        # D. Xử lý nền
        BG_COLOR = (192, 192, 192) 
        bg_image = np.zeros(frame.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        
        # E. Gộp ảnh 
        output_image = (frame * mask_3d + bg_image * (1 - mask_3d)).astype(np.uint8)

        # F. Lưu file
        file_name = f"{OUTPUT_DIR}/frame_{frame_id}.jpg"
        cv2.imwrite(file_name, output_image)
        
        return file_name

    except Exception as e:
        return f"LOI: {str(e)}"

# 3. SPARK MAIN 
def main():
    print("\n" + "="*50)
    print(" ĐANG CHẠY PHIÊN BẢN FINAL FIX ")
    print("="*50 + "\n")

    spark = SparkSession.builder \
        .appName("SparkBackgroundRemover") \
        .master("local[2]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Schema
    json_schema = StructType([
        StructField("id", LongType(), True),
        StructField("timestamp", DoubleType(), True),
        StructField("image_data", StringType(), True)
    ])

    remove_bg_udf = udf(process_frame_logic, StringType())

    # Đọc Stream
    print(" Đang kết nối tới Camera Server...")
    raw_df = spark.readStream \
        .format("socket") \
        .option("host", "localhost") \
        .option("port", 6100) \
        .load()

    parsed_df = raw_df.select(from_json(col("value"), json_schema).alias("data")).select("data.*")
    
    processed_df = parsed_df.withColumn("saved_path", remove_bg_udf(col("image_data"), col("id"))) \
                            .select("id", "timestamp", "saved_path")

    # Xuất ra màn hình
    query = processed_df.writeStream \
        .outputMode("append") \
        .format("console") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()