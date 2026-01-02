import socket
import cv2
import json
import base64
import time
import sys

# Cấu hình
HOST = "localhost"
PORT = 6100

def start_server():
    # 1. Khởi tạo Socket Server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    
    print(f" Camera Server đang chờ kết nối tại {HOST}:{PORT}...")
    conn, addr = s.accept()
    print(f" Đã kết nối với Spark Processor từ: {addr}")

    # 2. Mở Camera (Số 0 là webcam mặc định)
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        print(" Không thể mở Camera. Hãy kiểm tra lại!")
        sys.exit(1)

    try:
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize nhỏ lại (640x480) để gửi cho nhẹ
            frame = cv2.resize(frame, (640, 480))

            # 3. Mã hóa ảnh sang Base64
            # OpenCV encode sang jpg -> bytes -> base64 string
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # 4. Đóng gói JSON
            data = {
                'id': frame_id,
                'timestamp': time.time(),
                'image_data': jpg_as_text
            }
            
            # Gửi dữ liệu kèm ký tự xuống dòng \n để Spark biết hết 1 gói tin
            message = json.dumps(data) + "\n"
            conn.send(message.encode('utf-8'))
            
            print(f" Đã gửi frame {frame_id}")
            frame_id += 1
            
            # GIẢM TỐC ĐỘ GỬI 
            # Để 0.2 giây (tức 5 FPS) giúp Spark kịp xử lý mà không bị treo
            time.sleep(0.2) 

    except KeyboardInterrupt:
        print("\n Dừng server...")
    except BrokenPipeError:
        print("\n Kết nối bị ngắt đột ngột!")
    except Exception as e:
        print(f"\n Lỗi: {e}")
    finally:
        cap.release()
        conn.close()
        s.close()

if __name__ == "__main__":
    start_server()