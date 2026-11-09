from ultralytics import YOLO
import cv2


def main():
    # 1. 加载训练好的模型
    model = YOLO(r"C:\Users\木南\Desktop\ultralytics-main1\yolov8n.pt")

    # 2. 初始化笔记本摄像头
    cap = cv2.VideoCapture(0)  # 索引0表示默认摄像头

    # 3. 设置摄像头参数（可选）python realtime_webcam.py
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 高度

    try:
        while True:
            # 4. 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取摄像头画面！")
                break

            # 5. YOLOv8推理
            results = model.predict(
                frame,
                imgsz=320,  # 输入尺寸（越小越快，但精度可能下降）
                conf=0.5,  # 置信度阈值
                device="cpu"  # 使用CPU（若未安装GPU）
            )

            # 6. 绘制检测结果
            annotated_frame = results[0].plot()  # 自动绘制边界框和标签

            # 7. 显示实时画面
            cv2.imshow("YOLOv8 实时检测 - 按Q退出", annotated_frame)

            # 8. 检测退出按键
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # 9. 释放资源
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()