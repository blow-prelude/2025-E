import cv2
import numpy as np

bg = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.imshow("bg", bg)
cv2.waitKey(0)
cv2.destroyAllWindows()
