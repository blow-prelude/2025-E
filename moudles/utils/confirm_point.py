import cv2


def do_nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    img_width = 640
    img_height = 480
    window_name = "Image with Crosshair"
    cv2.namedWindow(window_name)

    # 设置滑动条的初始位置为图像中心
    cv2.createTrackbar("X Position", window_name, 0, img_width - 1, do_nothing)
    cv2.createTrackbar("Y Position", window_name, 0, img_height - 1, do_nothing)
    while True:
        (
            res,
            img,
        ) = cap.read()
        if not res:
            print("[INFO] camera read failed")
            continue
        # 从滑动条获取当前位置
        x_pos = cv2.getTrackbarPos("X Position", window_name)
        y_pos = cv2.getTrackbarPos("Y Position", window_name)

        # 创建一个原始图像的副本，这样我们就不会在原始图像上永久画线
        img_with_lines = img.copy()

        # 在副本上画线
        # 画水平线 (Horizontal line)
        # cv2.line(image, start_point, end_point, color, thickness)
        cv2.line(img_with_lines, (0, y_pos), (img_width, y_pos), (0, 255, 0), 2)  # 绿色

        # 画竖直线 (Vertical line)
        cv2.line(
            img_with_lines, (x_pos, 0), (x_pos, img_height), (0, 0, 255), 2
        )  # 红色

        # 显示带有线条的图像
        cv2.imshow(window_name, img_with_lines)

        # 等待按键，每 10 毫秒刷新一次
        key = cv2.waitKey(10) & 0xFF

        # 如果按下 'q' 键，则退出循环
        if key == ord("q"):
            break

        # 检查窗口是否被用户关闭
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    # --- 4. 清理 ---
    # 销毁所有创建的窗口
    cv2.destroyAllWindows()
    print("\n程序已退出。")


if __name__ == "__main__":
    main()
