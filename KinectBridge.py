import cv2
import numpy as np
import time
import socket
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

# --- 0. 解決 Python 3.8+ 移除 time.clock 的問題 ---
if not hasattr(time, 'clock'):
    time.clock = time.perf_counter

# --- 1. 初始化 Socket (發送給 Unity) ---
UDP_IP = "127.0.0.1"  # 若 Unity 在同一台電腦則維持不變
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- 2. 初始化 Kinect ---
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body)

# 這裡填入您校準得到的矩陣。若未校準，請維持 np.eye(3)
H = np.eye(3) 

def get_mapped_projection_coords(joint_x, joint_y, h_matrix):
    input_pt = np.array([[[joint_x, joint_y]]], dtype=np.float32)
    projected_pt = cv2.perspectiveTransform(input_pt, h_matrix)
    return projected_pt[0][0]

print(f"Kinect 已啟動，傳輸至 {UDP_IP}:{UDP_PORT}，等待偵測骨架...")

# 建立一個簡單的黑色視窗以便接收退出指令
cv2.namedWindow('Kinect Interaction Control')
dummy_img = np.zeros((100, 400, 3), np.uint8)
cv2.putText(dummy_img, "Press 'Q' to Quit", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

while True:
    cv2.imshow('Kinect Interaction Control', dummy_img)
    
    if kinect.has_new_body_frame():
        bodies = kinect.get_last_body_frame()

        if bodies is not None:
            for i in range(0, kinect.max_body_count):
                body = bodies.bodies[i]
                if not body.is_tracked:
                    continue

                joints = body.joints
                # 偵測右手 (HandRight)
                hand_right = joints[PyKinectV2.JointType_HandRight]

                if hand_right.TrackingState != PyKinectV2.TrackingState_NotTracked:
                    # A. 取得 Depth 空間座標 (這比原始公尺數值更穩定)
                    # 範圍通常為 X: 0~511, Y: 0~423
                    depth_pt = kinect.body_joint_to_depth_space(hand_right)
                    
                    # B. 歸一化座標 (0.0 ~ 1.0)
                    # 這樣 Unity 端不論解析度多少都能通用
                    norm_x = depth_pt.x / 512.0
                    norm_y = depth_pt.y / 424.0

                    # C. 執行 M2 座標映射 (如有 H 矩陣)
                    px, py = get_mapped_projection_coords(norm_x, norm_y, H)

                    # D. 輸出並發送數據給 Unity (M4)
                    message = f"{px:.4f},{py:.4f}"
                    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
                    
                    print(f"發送數據: {message}")

    # 檢查是否按下 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kinect.close()
cv2.destroyAllWindows()