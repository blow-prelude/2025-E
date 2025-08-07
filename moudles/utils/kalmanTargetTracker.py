from typing import List, Tuple

import cv2
import numpy as np


class KalmanTargetTracker:
    """
    使用卡尔曼滤波器来追踪和确认一个目标。
    """

    def __init__(
        self,
        dt=1,
        process_noise=1e-4,
        measurement_noise=1e-1,
        unseen_threshold=5,
        confirmation_threshold=3,
    ):
        """
        :param dt: 帧之间的时间间隔，通常为1
        :param process_noise: 过程噪声协方差，模型预测的不确定性。值越小，认为运动模型越准。
        :param measurement_noise: 测量噪声协方差，检测器（FindBox）的不确定性。值越小，认为检测结果越准。
        :param unseen_threshold: 连续多少帧未见到目标后，判定为丢失。
        :param confirmation_threshold: 需要连续多少帧更新后，才确认为稳定目标。
        """
        self.unseen_threshold = unseen_threshold
        self.confirmation_threshold = confirmation_threshold

        # 状态向量 [x, y, vx, vy] - 中心点位置和速度
        self.kf = cv2.KalmanFilter(4, 2)
        # 测量向量 [x, y]
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # 状态转移矩阵 A
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        # 过程噪声 Q
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        # 测量噪声 R
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        self.is_tracking = False
        self.is_confirmed = False
        self.frames_since_seen = 0
        self.frames_updated = 0
        self.last_box_info = None  # 存储最新的 (角点, h, w)

    def _get_center(self, box_corners: np.ndarray) -> np.ndarray:
        return np.mean(box_corners, axis=0)

    def update(self, candidates: List[Tuple[np.ndarray, float, float]]):
        """
        处理新一帧的候选者列表，并返回追踪状态。

        :param candidates: FindBox.get_a4_candidates() 的返回结果。
        :return: 如果目标已确认，返回目标信息(角点, h, w)。否则返回None。
        """
        # 1. 预测
        predicted_state = self.kf.predict()
        predicted_center = predicted_state[:2].reshape(2)

        # 2. 数据关联：在候选者中寻找最佳匹配
        best_match = None
        if candidates and self.is_tracking:
            min_dist = float("inf")
            # 寻找与预测位置最近的候选者
            for cand_info in candidates:
                cand_center = self._get_center(cand_info[0])
                dist = np.linalg.norm(cand_center - predicted_center)
                if dist < min_dist:
                    min_dist = dist
                    best_match = cand_info

        # 3. 更新
        # 如果找到了匹配项，用它来校正滤波器
        if best_match:
            self.frames_since_seen = 0
            self.frames_updated += 1
            measurement = self._get_center(best_match[0]).astype(np.float32)
            self.kf.correct(measurement)
            self.last_box_info = best_match
            print(
                f"[TRACKER] Updated. Consec_updates: {self.frames_updated}, Unseen: {self.frames_since_seen}"
            )
        # 如果没找到匹配，或者还没开始追踪
        else:
            self.frames_since_seen += 1
            print(
                f"[TRACKER] Not updated. Consec_updates: {self.frames_updated}, Unseen: {self.frames_since_seen}"
            )

            # 如果连续多帧没看到，重置追踪器
            if self.frames_since_seen > self.unseen_threshold:
                print("[TRACKER] Target lost. Resetting.")
                self.reset()
                # 丢失后，如果当前帧有候选者，可以立即开始新的追踪
                if candidates:
                    self._start_tracking(candidates[0])

            # 如果还未开始追踪，但有候选者，则开始追踪
            elif not self.is_tracking and candidates:
                self._start_tracking(candidates[0])

        # 4. 确认目标
        if self.frames_updated >= self.confirmation_threshold and not self.is_confirmed:
            self.is_confirmed = True
            print("\n!!! TARGET CONFIRMED !!!\n")

        if self.is_confirmed:
            # 返回一个由平滑后的状态和最新检测框组成的信息
            smooth_center = self.kf.statePost[:2].flatten()
            # 更新最新box的中心点为平滑后的点
            if self.last_box_info is not None:
                latest_corners = self.last_box_info[0]
            current_center = self._get_center(latest_corners)
            offset = smooth_center - current_center
            smooth_corners = latest_corners + offset

            if self.last_box_info is not None:
                return (smooth_corners, self.last_box_info[1], self.last_box_info[2])
            else:
                print(
                    "[TRACKER] Warning: last_box_info is None. Cannot return target info."
                )
                return None

        return None

    def _start_tracking(self, initial_box_info: Tuple[np.ndarray, float, float]):
        """用第一个检测到的目标来初始化卡尔曼滤波器"""
        print("[TRACKER] Starting new track.")
        center = self._get_center(initial_box_info[0])
        # 初始化状态向量
        self.kf.statePost = np.array([center[0], center[1], 0, 0], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self.is_tracking = True
        self.frames_updated = 1
        self.last_box_info = initial_box_info

    def reset(self):
        """重置追踪器状态"""
        self.is_tracking = False
        self.is_confirmed = False
        self.frames_since_seen = 0
        self.frames_updated = 0
        self.last_box_info = None
