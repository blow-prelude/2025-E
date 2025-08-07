from typing import Any, Tuple

import cv2
import numpy as np
from cv2.typing import MatLike

# typing hint
from numpy.typing import NDArray
from serSystem import SerSystem

import myglobal

from . import configs
from .utils.findBox import FindBox

# from .utils.targetConfirmer import TargetConfirmer
from .utils.kalmanTargetTracker import KalmanTargetTracker


def create_dummy_image(
    size=(480, 640),
    rect_pos=None,
    rect_size=(100, 141),  # 模拟高宽比为1.41的矩形
    noise_rect_pos=None,
    noise_rect_size=(50, 50),  # 模拟一个正方形干扰物
) -> MatLike:
    """创建一个带有矩形目标的模拟二值图"""
    img = np.zeros(size, dtype=np.uint8)
    if rect_pos:
        x, y = rect_pos
        w, h = rect_size
        cv2.rectangle(img, (x, y), (x + w, y + h), 255, -1)
    if noise_rect_pos:
        x, y = noise_rect_pos
        w, h = noise_rect_size
        cv2.rectangle(img, (x, y), (x + w, y + h), 255, -1)
    return img


class VideoSystem:
    def __init__(self):
        self.x = 0
        self.kernels = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # 初始化滤波器
        # self.tracker = KalmanTargetTracker(confirmation_threshold=3, unseen_threshold=5)
        self.tracker = KalmanTargetTracker()
        # 任务状态
        self.is_gimbal_rotating = True
        self.target_locked_and_data_sent = False
        # 初始化发送串口
        self.stm_ser = SerSystem(configs.stm_port, configs.baudrate, configs.timeout)
        self.cap = cv2.VideoCapture(configs.VIDEO_PATH)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, configs.VIDEO_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, configs.VIDEO_WIDTH)
        # 图像中心
        self.img_center: NDArray = np.array((204, 320))

    def q2_process(self):
        # 初始化ser对象
        # ser = SerSystem(configs.port, configs.baudrate, configs.timeout)
        is_target_finished = False

        try:
            while not is_target_finished:
                res, frame = self.cap.read()
                if not res:
                    print("[INFO] camera read failed")
                    return
                canvas = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                binarys = cv2.threshold(
                    gray, configs.bin_thres, configs.bin_maxval, cv2.THRESH_BINARY_INV
                )[1]

                opens = cv2.morphologyEx(
                    binarys, cv2.MORPH_OPEN, self.kernels, iterations=2
                )
                cv2.imshow("opens", opens)
                find_box = FindBox(opens)
                package = find_box.get_box()
                if len(package) != 0:
                    box_corner, target_h, target_w = package
                    print(f"[INFO] target height: {target_h}, target width: {target_w}")

                    # 图像中心点
                    cv2.circle(
                        canvas,
                        tuple(self.img_center),
                        radius=3,
                        color=(0, 255, 0),
                        thickness=3,
                    )

                    if box_corner.size != 0:
                        # 然后发送靶心和图像中心点坐标
                        target_center: NDArray = self.get_center_point(box_corner)
                        # int_target = target.astype(np.int32)
                        # 靶心坐标
                        cv2.circle(
                            canvas,
                            tuple(target_center),
                            radius=3,
                            color=(0, 0, 255),
                            thickness=3,
                        )
                        print(f"img center: {self.img_center}, target: {target_center}")

                        if (
                            abs(self.img_center[0] - target_center[0])
                            < configs.DISTANCE_DIFF
                            and abs(self.img_center[1] - target_center[1])
                            < configs.DISTANCE_DIFF
                        ):
                            print("[INFO] target is locked!")
                            # ser.sendTargetInfo(target_center, target_h, target_w)
                            is_target_finished = True

                        # 获得离散点
                        package = self.get_circle(target_center, target_h, target_w, 90)
                        if package != tuple():
                            discrete_points, radius = package
                            if discrete_points and len(discrete_points) > 0:
                                print(f"totally get points:{len(discrete_points)}")
                            cv2.circle(
                                canvas,
                                tuple(target_center),
                                radius=radius,
                                color=(255, 0, 0),
                                thickness=3,
                            )
                        # for point in discrete_points:
                        #     cv2.circle(canvas, point, 2, configs.DISCRETE_POINT_COLOR, -1)

                        else:
                            print("[WARN] discrete points is None!")
                cv2.imshow("box", canvas)

                if cv2.waitKey(1) & 0xFF == 27:
                    print("[INFO] exit by user")
                    break

        except KeyboardInterrupt:
            print("interupt by user,exit...")
            return

        finally:
            self.safe_exit()

            return

    def q2_process_new(self):
        try:
            res, frame = self.cap.read()
            if not res:
                print("[INFO] camera read failed")
                return
            canvas = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binarys = cv2.threshold(
                gray, configs.bin_thres, configs.bin_maxval, cv2.THRESH_BINARY_INV
            )[1]
            cv2.imshow("bins", binarys)
            # 找到所有获选者
            find_box = FindBox(binarys)
            candidates = find_box.get_a4_candidates()
            print(f"FindBox found {len(candidates)} candidate(s).")
            # 用候选者更新追踪器
            confirmed_target_info = self.tracker.update(candidates)

            if confirmed_target_info:
                # 提取目标信息
                smooth_corners, height, width = confirmed_target_info
                center = self.tracker._get_center(smooth_corners)
                int_center = [int(num) for num in center]
                lt, rt, rb, lb = smooth_corners
                cv2.line(canvas, tuple(lt), tuple(rt), (0, 255, 0), 2)
                cv2.line(canvas, tuple(rt), tuple(rb), (0, 255, 0), 2)
                cv2.line(canvas, tuple(rb), tuple(lb), (0, 255, 0), 2)
                cv2.line(canvas, tuple(lb), tuple(lt), (0, 255, 0), 2)
                cv2.circle(
                    canvas, tuple(int_center), radius=3, color=(0, 0, 255), thickness=3
                )
            cv2.imshow("canvas", canvas)
        finally:
            self.change_to_free()
            return

    def q2_process_copy(self):
        is_target_finished = False
        try:
            while not is_target_finished:
                res, frame = self.cap.read()
                if not res:
                    print("[INFO] camera read failed")
                    return
                canvas = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                binarys = cv2.threshold(
                    gray, configs.bin_thres, configs.bin_maxval, cv2.THRESH_BINARY_INV
                )[1]
                cv2.imshow("bins", binarys)
                closed = cv2.morphologyEx(
                    binarys,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50)),
                )
                cv2.imshow("closed", closed)
                contours_data = cv2.findContours(
                    closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = (
                    contours_data[1] if len(contours_data) == 3 else contours_data[0]
                )
                candidates = []
                for cnt in contours:
                    is_rect, approx = self.is_approx_rect(cnt)
                    if is_rect:
                        center = self.calc_center(approx)
                        if center:
                            candidates.append((approx, center, cv2.contourArea(approx)))
                if not candidates:
                    selected = None
                elif prev_center is None:
                    selected = max(candidates, key=lambda x: x[2])
                else:
                    candidates.sort(key=lambda x: self.distance(x[1], prev_center))
                    top_n = [candidates[0]]
                    for c in candidates[1:]:
                        if (
                            self.distance(c[1], prev_center)
                            - self.distance(candidates[0][1], prev_center)
                            < 50
                        ):
                            top_n.append(c)
                        else:
                            break
                    selected = max(top_n, key=lambda x: x[2])

                display_frame = frame.copy()
                contour_img = np.zeros_like(frame)

                if selected:
                    approx, center, _ = selected
                    cv2.drawContours(display_frame, [approx], -1, (0, 0, 255), 5)
                    cv2.circle(display_frame, center, 7, (0, 0, 255), -1)
                    cv2.drawContours(contour_img, [approx], -1, (0, 255, 0), 3)
                    prev_center = center
                else:
                    prev_center = None

        except Exception as e:
            print(f"[ERROR] Exception in q2_process: {e}")
        finally:
            self.change_to_free()
            print("[TASK] Q2 process finished.")

    def is_approx_rect(self, contour, epsilon_factor=0.02):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
        return (4 <= len(approx) <= 5 and cv2.isContourConvex(approx)), approx

    def calc_center(self, approx):
        M = cv2.moments(approx)
        if M["m00"] == 0:
            return None
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def is_approx_rect(self, contour, epsilon_factor=0.02):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
        return (4 <= len(approx) <= 5 and cv2.isContourConvex(approx)), approx

    def calc_center(self, approx):
        M = cv2.moments(approx)
        if M["m00"] == 0:
            return None
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def q3_process(self):
        """
        在扫描阶段，如果连续5次都’找到‘了靶心，就认为这个是靶心
        然后和第二问相同处理"""
        try:
            res, frame = self.cap.read()
            if not res:
                print("[INFO] camera read failed")
                return
            canvas = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binarys = cv2.threshold(
                gray, configs.bin_thres, configs.bin_maxval, cv2.THRESH_BINARY_INV
            )[1]

            opens = cv2.morphologyEx(
                binarys, cv2.MORPH_OPEN, self.kernels, iterations=3
            )
            cv2.imshow("opens", opens)
            if not self.target_locked_and_data_sent:
                # 找到所有潜在候选者
                find_box = FindBox(opens)
                candidates = find_box.get_a4_candidates()
                print(f"FindBox found {len(candidates)} candidate(s).")

                # 用候选者更新追踪器
                confirmed_target_info = self.tracker.update(candidates)

                if confirmed_target_info:
                    # 提取目标信息
                    smooth_corners, height, width = confirmed_target_info
                    center = self.tracker._get_center(smooth_corners)
                    print(
                        f">>> TARGET LOCKED! Smoothed Center: ({center[0]:.1f}, {center[1]:.1f})"
                    )
                    # 发送串口消息
                    # ser

                    self.target_locked_and_data_sent = True
                else:
                    print("has lock the target")
                    self.q2_process()
            # package = find_box.get_box()
            # if len(package) != 0:
        finally:
            self.safe_exit()
            return

    def q4_process(self):
        # 启动接收串口
        ti_ser = SerSystem(configs.ti_port, configs.baudrate, configs.timeout)
        is_target_finished = False
        try:
            while not is_target_finished:
                ti_receive = ti_ser.receiveSigns()
                # 接收到停止指令，退出任务
                if ti_receive == "n":
                    is_target_finished = True
                    myglobal.status = 0
                res, frame = self.cap.read()
                if not res:
                    print("[INFO] camera read failed")
                    return
                canvas = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                binarys = cv2.threshold(
                    gray, configs.bin_thres, configs.bin_maxval, cv2.THRESH_BINARY_INV
                )[1]

                opens = cv2.morphologyEx(
                    binarys, cv2.MORPH_OPEN, self.kernels, iterations=2
                )
                cv2.imshow("opens", opens)
                find_box = FindBox(opens)
                package = find_box.get_box()
                if len(package) != 0:
                    box_corner, target_h, target_w = package
                    print(f"[INFO] target height: {target_h}, target width: {target_w}")

                    # 图像中心点
                    cv2.circle(
                        canvas,
                        tuple(self.img_center),
                        radius=3,
                        color=(0, 255, 0),
                        thickness=3,
                    )

                    if box_corner.size != 0:
                        # 然后发送靶心和图像中心点坐标
                        target_center: NDArray = self.get_center_point(box_corner)
                        # int_target = target.astype(np.int32)
                        # 靶心坐标
                        cv2.circle(
                            canvas,
                            tuple(target_center),
                            radius=3,
                            color=(0, 0, 255),
                            thickness=3,
                        )
                        print(f"img center: {self.img_center}, target: {target_center}")

        finally:
            self.safe_exit()
            return

    def get_center_point(self, box_corner: NDArray) -> NDArray:
        """
        param box_corner  4个角点坐标
        :return  中心点坐标
        """
        if box_corner.shape[0] != 4:
            print("[WARNIMG] you must provide 4 points!")
            return np.empty(0, dtype=np.float32)
        # center_x = np.mean(box_corner[:, 0])
        # center_y = np.mean(box_corner[:, 1])
        line1 = (box_corner[0], box_corner[2])
        line2 = (box_corner[1], box_corner[3])
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            # 平行线无交点
            return np.empty(0, dtype=np.float32)
        d = (det(*line1), det(*line2))
        center_x = det(d, xdiff) / div
        center_y = det(d, ydiff) / div
        return np.array((center_x, center_y), dtype=np.int32)

    def get_circle(
        self, target_center: NDArray, target_height, target_width, num_points: int
    ):
        """
        :param target_center: NDArray   靶心坐标
        :param num_points: int  生成的离散点的数量
        """
        if target_center.size == 0:
            print("target center is empty by accident!")
            return tuple()
        target_x, target_y = target_center
        radius = self.get_virtual_radius(target_height, target_width)
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x_coords = target_x + radius * np.cos(angles)
        y_coords = target_y + radius * np.sin(angles)
        # 转换为整数类型
        points = np.round(np.column_stack((x_coords, y_coords))).astype(int)
        return ([tuple(p) for p in points], radius)

    def get_virtual_radius(self, virtual_height, virtual_width):
        """
        return virtual_radius: int
        返回当前画面上的圆形半径
        """
        actual_radius = 60
        actual_width = 297
        actual_height = 210
        # 计算缩放比例
        ration1 = virtual_width / actual_width
        ration2 = virtual_height / actual_height
        ration = (ration1 + ration2) / 2
        virtual_radius = int(actual_radius * ration)
        return virtual_radius

    def myPerspectiveTransform(
        self, origin: MatLike, pt1: NDArray, is_resize=False
    ) -> Any:
        """
        :param origin   原始图像
        :param pt1   透视变换前的4个角点坐标
        :param is_resize    是否调整尺寸，默认不进行
        return (dest,origin_w,origin_h)  分别是 透视变换后的图，轮廓的宽，轮廓的高"""
        if len(pt1) == 0:
            return
        (lt, rt, rb, lb) = pt1
        w1 = np.sqrt((lt[0] - rt[0]) ** 2 + (lt[1] - rt[1]) ** 2)
        w2 = np.sqrt((lb[0] - rb[0]) ** 2 + (lb[1] - rb[1]) ** 2)
        origin_w = max(int(w1), int(w2))
        h1 = np.sqrt((lt[0] - lb[0]) ** 2 + (lt[1] - lb[1]) ** 2)
        h2 = np.sqrt((rt[0] - rb[0]) ** 2 + (rt[1] - rb[1]) ** 2)
        origin_h = max(int(h1), int(h2))
        pt2 = np.array(
            [[0, 0], [origin_w, 0], [origin_w, origin_h], [0, origin_h]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(pt1, pt2)
        dest: MatLike = cv2.warpPerspective(origin, M, (origin_w, origin_h))
        if is_resize:
            h, w = origin.shape[:2]
            dest = cv2.resize(dest, (w, h))
        return (dest, origin_w, origin_h)

    def safe_exit(self):
        self.cap.release()
        self.stm_ser.close()
        cv2.destroyAllWindows()
        print("[INFO] video system closed.")

    def change_to_free(self):
        myglobal.status = 0
        print("[INFO] change to mode 0")


class MockCamera:
    def __init__(self, img_size=(configs.IMG_WIDTH, configs.IMG_HEIGHT)):
        self.img_size = img_size
        self.width, self.height = img_size
        self.frame_count = 0
        # 目标A4纸，尺寸为120x170，初始位置在画面外右侧
        self.target_pos = [self.width, self.height // 2 - 85]
        self.target_size = (120, int(120 * configs.A4_ASPECT_RATIO))
        # 云台逆时针转动，等效于景物向右移动
        self.pan_speed = 30  # pixels per frame

    def get_frame(self) -> Tuple[MatLike, MatLike]:
        self.frame_count += 1
        # 更新目标位置来模拟云台转动
        self.target_pos[0] -= self.pan_speed

        # 创建彩色图和二值图
        color_frame = np.full(
            (self.height, self.width, 3), (20, 20, 20), dtype=np.uint8
        )
        bin_frame = np.zeros((self.height, self.width), dtype=np.uint8)

        # 绘制目标
        x, y = int(self.target_pos[0]), int(self.target_pos[1])
        w, h = self.target_size
        if x < self.width and x + w > 0:  # 仅当目标在视野内时绘制
            cv2.rectangle(color_frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
            cv2.rectangle(bin_frame, (x, y), (x + w, y + h), 255, -1)

        return color_frame, bin_frame
