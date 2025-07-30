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
        self.cap = cv2.VideoCapture(configs.VIDEO_PATH)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, configs.VIDEO_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, configs.VIDEO_WIDTH)
        # 图像中心
        self.img_center: NDArray = self.get_center_point(
            np.array(
                (
                    (0, 0),
                    (configs.VIDEO_WIDTH, 0),
                    (configs.VIDEO_WIDTH, configs.VIDEO_HEIGHT),
                    (0, configs.VIDEO_HEIGHT),
                )
            )
        )

    def process(self):
        try:
            while True:
                res, frame = self.cap.read()
                if not res:
                    print("[INFO] camera read failed")
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                binarys = cv2.threshold(
                    gray, configs.bin_thres, configs.bin_maxval, cv2.THRESH_BINARY_INV
                )[1]
                find_box = FindBox(binarys)
                box_corner: NDArray = find_box.get_box()
                canvas = frame.copy()
                if box_corner.size != 0:
                    # 可视化一下
                    canvas, transform_w, transform_h = self.myPerspectiveTransform(
                        canvas, box_corner
                    )
                    # find_box.draw_box(box_corner, canvas)

                    # 发送停止扫描的信号，然后发送靶心和图像中心点坐标
                    target: NDArray = self.get_center_point(box_corner)
                    img_h, img_w = frame.shape[:2]

                    print(f"img center: {self.img_center}, target: {target}")
                cv2.imshow("box", canvas)
                x = 1
                if cv2.waitKey(1) & 0xFF == 27:
                    print("[INFO] exit by user")
                    break
                elif cv2.waitKey(1) & 0xFF == ord("s"):
                    cv2.imwrite(f"img{x}", canvas)
                    x += 1

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
            return np.empty(0)
        center_x = np.mean(box_corner[:, 0])
        center_y = np.mean(box_corner[:, 1])
        return np.array((center_x, center_y), dtype=np.float32)

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
