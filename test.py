import cv2

# from numpy.typing import NDArray
import numpy as np

# typing hint
import moudles.configs as configs
import moudles.utils.utils_configs as utils_configs

# from moudles.utils.findBox import FindBox


def main():
    img = cv2.imread("img1.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binarys = cv2.threshold(
        gray, configs.bin_thres, configs.bin_maxval, cv2.THRESH_BINARY_INV
    )[1]
    kernels = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opens = cv2.morphologyEx(binarys, cv2.MORPH_OPEN, kernels, iterations=3)
    cv2.imshow("opens", opens)
    cons = cv2.findContours(opens, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(cons) == 0:
        return
    biggest_con = max(cons, key=cv2.contourArea)
    canvas = img.copy()
    # """
    peri = cv2.arcLength(biggest_con, True)
    approx = cv2.approxPolyDP(biggest_con, peri * utils_configs.EPSILON_RATION, True)
    print(len(approx))

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
        box_corner = np.array(
            [left_top, right_top, right_bottom, left_bottom],
            dtype=np.int32,
        ).reshape(4, 2)
        if len(box_corner) == 4:
            mycolor = (0, 0, 255)
            # 转化成整数元组
            lt, rt, rb, lb = [tuple(map(int, pt)) for pt in box_corner]
            cv2.line(canvas, lt, rt, mycolor, 2)
            cv2.line(canvas, rt, rb, mycolor, 2)
            cv2.line(canvas, rb, lb, mycolor, 2)
            cv2.line(canvas, lb, lt, mycolor, 2)
    # """
    # cv2.drawContours(canvas, [biggest_con], -1, (0, 255, 0), 3)
    # find_box = FindBox(binarys)
    # box_corner: NDArray = find_box.get_box()[0]
    # find_box.draw_box(box_corner, canvas)
    # cv2.imshow("img", canvas)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
