from typing import Sequence, Tuple

import cv2
import numpy as np

# typing hint
from cv2.typing import MatLike
from numpy.typing import NDArray

from . import utils_configs as configs


class FindBox:
    """
    parma bin_img  二值化处理后的图像,只有一个通道
    :"""

    def __init__(self, bin_img: MatLike):
        self.bin_img = bin_img

    def get_box(self) -> Tuple[NDArray, float, float]:
        """
        return  外框4个角点坐标
        : 该函数可以返回图像中最大轮廓的4个角点
        """
        # bin_img 必须是单通道图像
        if self.bin_img.ndim == 3:
            raise Exception("bin_img must be a 2D array")
        contours: Sequence = cv2.findContours(
            self.bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        if len(contours) == 0:
            print("[WARMING] no contours found")
            return tuple()
        package = self.confirm_box(contours)
        if package == tuple():
            return tuple()
        box_corner, target_h, target_w = package
        return (box_corner, target_h, target_w)
        # np method
        # time1 = time()
        # areas = np.array([cv2.contourArea(c) for c in contours])
        # max_contour_idx = np.argmax(areas)
        # biggest_contour = contours[max_contour_idx]
        # time2 = time()
        # print(f"[FREQENCY] numpy method use time {time2 - time1:6f}")

        # box_sort: NDArray[np.float32] = np.zeros((4, 2), dtype=np.float32)
        # 按照左上右上的顺时针排序
        # print(f"[DEBUG] sorted box is {box_corner}")

    def confirm_box(self, contours: Sequence[MatLike]) -> Tuple[NDArray, float, float]:
        """
        : param contours   所有轮廓
        : return certain_box   确定的A4纸角点
        输入所有的轮廓，返回靶纸的4个角点
        """
        for idx, con in enumerate(contours):
            area = cv2.contourArea(con)
            if area > configs.MINAREA:
                peri = cv2.arcLength(con, True)
                approx = cv2.approxPolyDP(con, peri * configs.EPSILON_RATION, True)
                if len(approx) == 4:
                    approx = np.array(approx, dtype=np.float32).reshape((4, 2))
                    # print(f"approx : {approx}")
                    # 先按x再按y排列，那么前2个是左上和左下
                    box = sorted(approx, key=lambda pt: (pt[0], pt[1]))
                    if box[0][1] > box[1][1]:
                        left_bottom, left_top = box[0], box[1]
                    else:
                        left_bottom, left_top = box[1], box[0]
                    if box[2][1] > box[3][1]:
                        right_bottom, right_top = box[2], box[3]
                    else:
                        right_bottom, right_top = box[3], box[2]

                    # 计算宽高之比
                    w1 = np.sqrt(
                        (left_top[0] - right_top[0]) ** 2
                        + (left_top[1] - right_top[1]) ** 2
                    )
                    w2 = np.sqrt(
                        (left_bottom[0] - right_bottom[0]) ** 2
                        + (left_bottom[1] - right_bottom[1]) ** 2
                    )
                    h1 = np.sqrt(
                        (left_top[0] - left_bottom[0]) ** 2
                        + (left_top[1] - left_bottom[1]) ** 2
                    )
                    h2 = np.sqrt(
                        (right_top[0] - right_bottom[0]) ** 2
                        + (right_top[1] - right_bottom[1]) ** 2
                    )
                    avarage_h = (h1 + h2) / 2
                    avarage_w = (w1 + w2) / 2
                    h_w_rate = avarage_h / avarage_w * 1.0
                    if (
                        configs.WIDTH_HEIGHT_RATION_HIGH
                        >= h_w_rate
                        >= configs.WIDTH_HEIGHT_RATION_LOW
                    ):
                        print("[INFO] successfully find certain box")
                        print(f"area of con : {area}")
                        print(f"h_w_rate: {h_w_rate}")

                        return (
                            np.array(
                                [left_top, right_top, right_bottom, left_bottom],
                                dtype=np.float32,
                            ).reshape(4, 2),
                            avarage_h,
                            avarage_w,
                        )
                    else:
                        # print("[INFO] cannot find a A4 ractangle")
                        # print(f"area of con : {area}")
                        # print(f"h_w_rate: {h_w_rate}")
                        continue
        # print("find no suitable box")
        return tuple()

    def draw_box(self, box_corner: NDArray, canvas: MatLike) -> None:
        """
        param box_corner  4个角点的坐标
        param canvas  画布
        """
        mycolor = (0, 0, 255)
        if canvas.ndim == 3:
            print(f"box corner : {box_corner}")
            if len(box_corner) == 4:
                # 转化成整数元组
                lt, rt, rb, lb = [tuple(map(int, pt)) for pt in box_corner]
                cv2.line(canvas, lt, rt, mycolor, 2)
                cv2.line(canvas, rt, rb, mycolor, 2)
                cv2.line(canvas, rb, lb, mycolor, 2)
                cv2.line(canvas, lb, lt, mycolor, 2)

            elif len(box_corner) == 0:
                print("[ERROR] box corner is empty by accident!")
            else:
                print("[ERROR] the shape of box corner is changed by accident!")
        else:
            print("your canvas is not a RGB image")
