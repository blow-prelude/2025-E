from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


class TargetConfirmer:
    """
    通过连续帧的候选列表来确认一个稳定的目标。
    """

    def __init__(
        self,
        confirmation_threshold: int = 3,
        max_distance_threshold: float = 100.0,
        image_center: Tuple[int, int] = (0, 0),
    ):
        """
        :param confirmation_threshold: 连续识别多少次后确认为目标。
        :param max_distance_threshold: 判断是否为同一目标的最大中心点距离（像素）。
                                       这个值需要根据你的云台移动速度和帧率来调整。
                                       移动越快，这个值需要越大。
        :param image_center: 图像中心点坐标 (cx, cy)，用于在有多个新目标时选择一个开始追踪。
        """
        self.confirmation_threshold = confirmation_threshold
        self.max_distance_threshold = max_distance_threshold
        self.image_center = image_center

        self.candidate_box_info = None  # 存储当前正在追踪的候选目标 (角点, h, w)
        self.confirmation_count = 0
        self.confirmed_target = None

        print(
            f"TargetConfirmer initialized: Need {confirmation_threshold} consecutive frames to confirm."
        )
        print(f"Max distance for same target: {max_distance_threshold} pixels.")

    def _get_center(self, box_corners: NDArray) -> NDArray:
        """计算轮廓的中心点"""
        return np.mean(box_corners, axis=0)

    def process_frame(self, new_candidates: List[Tuple[NDArray, float, float]]):
        """
        处理新一帧的候选者列表。

        :param new_candidates: FindBox.get_a4_candidates() 的返回结果（一个列表）。
        :return: 如果目标已确认，返回确认的目标信息 (角点, h, w)；否则返回 None。
        """
        if self.confirmed_target:
            return self.confirmed_target

        # 1. 当前帧没有检测到任何候选者
        if not new_candidates:
            if self.candidate_box_info:
                print("[STATUS] Candidate lost. Resetting confirmation.")
                self.reset()
            return None

        # 2. 之前没有追踪任何候选者
        if self.candidate_box_info is None:
            # 从新候选者中选择一个开始追踪
            # 优先选择最靠近图像中心的那个
            if self.image_center:
                new_candidates.sort(
                    key=lambda c: np.linalg.norm(
                        self._get_center(c[0]) - self.image_center
                    )
                )

            self.candidate_box_info = new_candidates[0]
            self.confirmation_count = 1
            print(
                f"[STATUS] New candidate acquired at center {self._get_center(self.candidate_box_info[0])}. Starting confirmation."
            )
            return None

        # 3. 之前有追踪的候选者，现在需要从新候选者中找到它
        candidate_center = self._get_center(self.candidate_box_info[0])

        best_match = None
        min_dist = float("inf")

        for new_cand_info in new_candidates:
            dist = np.linalg.norm(self._get_center(new_cand_info[0]) - candidate_center)
            if dist < min_dist:
                min_dist = dist
                best_match = new_cand_info

        # 3.1. 找到了一个足够近的匹配项
        if min_dist < self.max_distance_threshold:
            self.confirmation_count += 1
            self.candidate_box_info = best_match  # 更新追踪目标为最新位置
            print(
                f"[STATUS] Candidate re-identified. Count: {self.confirmation_count}/{self.confirmation_threshold}. Distance: {min_dist:.2f}px"
            )

            # 检查是否达到确认阈值
            if self.confirmation_count >= self.confirmation_threshold:
                self.confirmed_target = self.candidate_box_info
                if self.confirmed_target is not None:
                    print(
                        f"\n!!! TARGET CONFIRMED at center {self._get_center(self.confirmed_target[0])} !!!\n"
                    )
                else:
                    print("\n!!! TARGET CONFIRMED but target data is missing !!!\n")
                return self.confirmed_target

        # 3.2. 所有新候选者都离得太远，认为追踪目标已丢失
        else:
            print(
                f"[STATUS] Tracked candidate lost (min distance to new candidates was {min_dist:.2f}px). Resetting."
            )
            self.reset()
            # 立即尝试从当前帧的候选者中开始新的追踪
            # 递归调用一次，以处理当前帧的新候选者
            return self.process_frame(new_candidates)

        return None

    def reset(self):
        """重置状态，用于目标丢失或重新搜索。"""
        self.candidate_box_info = None
        self.confirmation_count = 0
        self.confirmed_target = None
        print("[STATUS] Confirmer has been reset.")
