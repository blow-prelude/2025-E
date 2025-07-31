from typing import Any

import cv2
import numpy as np
from cv2.typing import MatLike

# typing hint
from numpy.typing import NDArray

from . import configs
from .utils.findBox import FindBox


class VideoSystem:
    def __init__(self):
        self.x = 0
        self.kernels = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.cap = cv2.VideoCapture(configs.VIDEO_PATH)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, configs.VIDEO_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, configs.VIDEO_WIDTH)
        # 图像中心
        self.img_center: NDArray = np.array((204, 320))

    def process(self):
        # 初始化ser对象
        # ser = SerSystem(configs.port, configs.baudrate, configs.timeout)
        try:
            while True:
                res, frame = self.cap.read()
                if not res:
                    print("[INFO] camera read failed")
                    continue
                canvas = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                binarys = cv2.threshold(
                    gray, configs.bin_thres, configs.bin_maxval, cv2.THRESH_BINARY_INV
                )[1]

                opens = cv2.morphologyEx(
                    binarys, cv2.MORPH_OPEN, self.kernels, iterations=3
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
                        # 发送停止扫描的信号，然后发送靶心和图像中心点坐标
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
                elif cv2.waitKey(1) & 0xFF == ord("s"):
                    cv2.imwrite(f"img{self.x}.jpg", canvas)
                    print(f"write into img{self.x}")
                    self.x += 1

        except KeyboardInterrupt:
            print("interupt by user,exit...")

        finally:
            self.safe_exit()

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
        cv2.destroyAllWindows()
        print("[INFO] video system closed.")
