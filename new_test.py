import cv2
import numpy as np
import serial


def sendPoints(ser, data):
    """
    data: Sequence  2个整数坐标，先当前位置，在目标位置
    发送 #开头， ：结尾的数据
    """
    if ser.is_open and len(data) == 4:
        # payload = ",".join(map(str, data))
        # message = f":{payload}#"
        # ser.write(message.encode())
        ser.write(":".encode)
        ser.write(str(data[0]).encode())
        ser.write(",".encode())
        ser.write(str(data[1]).encode())
        ser.write(",".encode())
        ser.write(str(data[2]).encode())
        ser.write(",".encode())
        ser.write(str(data[3]).encode())
        ser.write(",".encode())
        ser.write("#".encode())

        print(f"Sent: {data}")
    else:
        print(f"data is not fit ,len is {len(data)}")


def receiveSigns(ser):
    """接收数据，解析n或者y"""
    if not ser.is_open:
        print("Serial port is not open.")
        return None
    if ser.in_waiting > 0:
        print("yes")
        line = ser.readline(ser.in_waiting)
        data_str = line.decode("utf-8").strip()
        if data_str in ("y2", "y3", "y4", "y5"):
            print(f"successfully receive {data_str}")
            return data_str
        elif data_str == "n":
            print(f"successfully receive {data_str}")
            return data_str
        elif data_str:
            # 接收到的不为空
            print(f"receive unexpected data :{data_str}")
            return None


def is_approx_rect(contour, epsilon_factor=0.02):
    """
    return  (是否是四边形，拟合多边形)
    判断是否是四边形"""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
    return (4 <= len(approx) <= 5 and cv2.isContourConvex(approx)), approx


def calc_center(approx):
    """
    return  质心整数坐标
    用拟合的多边形计算出质心坐标
    """
    M = cv2.moments(approx)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def distance(p1, p2):
    """
    两点之间距离计算公式
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("fail to open camera")
        return
    print("success open camera")
    # 初始化串口
    try:
        stm_ser = serial.Serial("/dev/ttyAMA3", 115200, timeout=0.01)
        ti_ser = serial.Serial("/dev/ttyAMA2", 115200, timeout=0.01)
        print("open stm_ser ti_ser successfully")
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")

    prev_center = None
    mode = 0
    # 当前坐标点（光斑坐标）
    current_x = 320
    current_y = 240
    MIN_DISTANCE = 20
    try:
        while True:
            sign_received = receiveSigns(stm_ser)
            if sign_received is not None:
                if sign_received == "n":
                    mode = 0
                if sign_received in ("y2", "y3", "y4", "y5"):
                    digit = int(sign_received[1])
                    mode = digit - 1

            if mode == 0:
                continue
            elif mode == 1:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
                closed = cv2.morphologyEx(
                    binary,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50)),
                )

                contours = cv2.findContours(
                    closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )[0]

                candidates = []
                for cnt in contours:
                    # 用四边形拟合，符合者加入候选名单
                    is_rect, approx = is_approx_rect(cnt)
                    if is_rect:
                        center = calc_center(approx)
                        # 计算质心
                        if center:
                            candidates.append((approx, center, cv2.contourArea(approx)))
                # 分为3种情况：没有候选者、当前第一帧找到候选者（或者之前跟丢了）、上一帧找到候选者
                if not candidates:
                    selected = None
                elif prev_center is None:
                    # 选择面积最大的作为候选者
                    selected = max(candidates, key=lambda x: x[2])
                else:
                    # 按“与上一帧中心点的距离”对所有候选者进行排序，最近的排在最前面
                    candidates.sort(key=lambda x: distance(x[1], prev_center))
                    top_n = [candidates[0]]
                    # 遍历其他候选者，如果它们和“最近者”的距离差在50像素以内，也加入优选组
                    for c in candidates[1:]:
                        if (
                            distance(c[1], prev_center)
                            - distance(candidates[0][1], prev_center)
                            < 50
                        ):
                            top_n.append(c)
                        else:
                            break
                    # 从这个“优选组”中，选出面积最大的那个作为本帧的目标
                    selected = max(top_n, key=lambda x: x[2])

                display_frame = frame.copy()
                contour_img = np.zeros_like(frame)

                # 有追踪目标时，更新目标的中点
                if selected:
                    approx, center, _ = selected
                    cv2.drawContours(display_frame, [approx], -1, (0, 0, 255), 5)
                    cv2.circle(display_frame, center, 7, (0, 0, 255), -1)
                    cv2.drawContours(contour_img, [approx], -1, (0, 255, 0), 3)
                    # 更新追踪目标中点坐标，下一帧就在这个点附近寻找追踪目标
                    prev_center = center
                    print(f"perv_center : {prev_center}")
                    prev_center_x, prev_center_y = prev_center
                    sendPoints(
                        stm_ser, (current_x, current_y, prev_center_x, prev_center_y)
                    )

                    current_point = (current_x, current_y)
                    prev_center = (prev_center_x, prev_center_y)
                    if distance(current_point, prev_center) <= MIN_DISTANCE:
                        pass

                else:
                    prev_center = None

                # cv2.imshow("display_frame", display_frame)
                # cv2.imshow("binary", binary)
                # cv2.imshow("closed", closed)
                # cv2.imshow("contour_img", contour_img)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        stm_ser.close()
        ti_ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
