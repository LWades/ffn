from symmetric_syndrome import SymmetricSyndrome
from toric_code import draw_toric_code
import numpy as np
from utils import log


class ReflectionSymmetricSyndrome(SymmetricSyndrome):
    """
    axis ->
        0: 垂直轴
        1: 水平轴
        2: 左 45 度
        3: 右 45 度
    """
    def __init__(self, d, syndrome, s_type='rf', axis=0):
        super().__init__(d, syndrome, s_type)
        self.axis = axis

    def base_syndrome(self):
        log(f"self.center = {self.center}")
        b_s = np.zeros(self.syndrome.shape)     # base_syndrome
        if self.axis == 0:
            axis_symmetry_y = self.center[1]
            for i in range(self.xs.shape[0]):
                x, y = self.xs[i], self.ys[i]
                y = (2 * axis_symmetry_y - y) % (2 * self.d)    # 周期性边界下的垂直轴对称
                b_s[x, y] = 1
        elif self.axis == 1:
            axis_symmetry_x = self.center[0]
            for i in range(self.xs.shape[0]):
                x, y = self.xs[i], self.ys[i]
                x = (2 * axis_symmetry_x - x) % (2 * self.d)
                b_s[x, y] = 1
        elif self.axis == 2:
            """
            set: center = (cx, cy)
            k = (cx - cy) / 2
            give (a, b) -> (cy + k, cx - k)[mod 2d]
            核心思量：沿着中心对称轴距离主对角线的距离先移动一段距离 |k|，坐标交换后返回
            """
            offset_center2xey_sub_times2 = self.center[0] - self.center[1]
            log(f"offset_center2xey_sub_times2 = {offset_center2xey_sub_times2}")
            for i in range(self.xs.shape[0]):
                x, y = self.xs[i], self.ys[i]
                b_s[(y + offset_center2xey_sub_times2) % (2 * self.d), (2 * self.d + x - offset_center2xey_sub_times2) % (2 * self.d)] = 1
                log(f"({x}, {y})->({y + offset_center2xey_sub_times2}, {x - offset_center2xey_sub_times2})")
        elif self.axis == 3:
            """
            set: center = (cx, cy)
            k = (cx + cy - (2d-1)) / 2
            give (a, b) -> ((2d-1) - b + 2k), (2d-1) - a + 2k)
            核心思想：沿着中心对称轴距离副对角线的距离先移动一段距离 |k|，坐标交换后用 2d-1 减，之后返回去
            """
            log(f"self.center = {self.center}")
            offset_center2xpyed_sub_times2 = self.center[0] + self.center[1] - (2 * self.d - 1)
            log(f"offset_center2xey_sub_times2 = {offset_center2xpyed_sub_times2}")
            for i in range(self.xs.shape[0]):
                x, y = self.xs[i], self.ys[i]
                b_s[(2 * self.d - 1 - y + offset_center2xpyed_sub_times2) % (2 * self.d), (2 * self.d - 1 - x + offset_center2xpyed_sub_times2) % (2 * self.d)] = 1
                log(f"({x}, {y})->({(2 * self.d - 1 - y + offset_center2xpyed_sub_times2) % (2 * self.d)}, {(2 * self.d - 1 - x + offset_center2xpyed_sub_times2) % (2 * self.d)})")
        else:
            log("Illigal axis")
        return b_s


if __name__ == '__main__':
    d = 3
    # syndrome = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    syndrome = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0])
    symmetric_syndrome = ReflectionSymmetricSyndrome(d, syndrome, axis=2)
    draw_toric_code(d, symmetric_syndrome, center=True)
    symmetric_syndrome.syndrome = symmetric_syndrome.base_syndrome()
    draw_toric_code(d, symmetric_syndrome, center=True)
