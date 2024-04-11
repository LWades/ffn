import numpy as np
import hashlib
from toric_code import draw_toric_code
from utils import log, ZX2image, ZX2image_full


class SymmetricSyndrome:
    def __init__(self, d, syndrome, s_types):
        if len(s_types) == 0:
            log("error: no s_types")
            exit(0)
        self.origin_syndrome = syndrome
        self.syndrome = ZX2image(d, syndrome)  # [......] -> [[...], [...], ..., [...]]
        self.s_types = s_types
        self.d = d
        self.xs = np.where(self.syndrome == 1)[0]
        self.ys = np.where(self.syndrome == 1)[1]
        self.center = self.get_center()     # (x_center, y_center)
        self.center_img = np.array([self.d - 1, self.d])

        # if 'rf' in self.s_types:
        #     self.axis = 0

    # 每个错误症状都有一个中心
    def get_center(self):
        if self.xs.size == 0 and self.ys.size == 0:
            return -1, -1       # 没有错误症状的元素没有中心
        center_udlr = (int(np.mean(self.xs)), int(np.mean(self.ys)))  # 为了效率小数部分直接截掉
        d_unlr = np.sum((self.xs - center_udlr[0]) ** 2 + (self.ys - center_udlr[1]) ** 2)

        xs_t = (self.xs + self.d) % (2 * self.d)
        mmlr_center_x_t = int(np.mean(xs_t))
        mmlr_center_x = (mmlr_center_x_t - self.d) % (2 * self.d)
        center_mmlr = (mmlr_center_x, int(np.mean(self.ys)))
        d_mmlr = np.sum((xs_t - mmlr_center_x_t) ** 2 + (self.ys - center_mmlr[1]) ** 2)

        ys_t = (self.ys + self.d) % (2 * self.d)
        udmm_center_y_t = int(np.mean(ys_t))
        udmm_center_y = (udmm_center_y_t - self.d) % (2 * self.d)
        center_udmm = (int(np.mean(self.xs)), udmm_center_y)
        d_udmm = np.sum((self.xs - center_udmm[0]) ** 2 + (ys_t - udmm_center_y_t) ** 2)

        center_mmmm = (mmlr_center_x, udmm_center_y)
        d_mmmm = np.sum((xs_t - mmlr_center_x_t) ** 2 + (ys_t - udmm_center_y_t) ** 2)
        # log("four centers：udlr -", center_udlr, "mmlr -", center_mmlr, "udmm -", center_udmm, "mmmm -", center_mmmm)
        # log("到四个中心的距离: udlr -", d_unlr, "mmlr -", d_mmlr, "udmm -", d_udmm, "mmmm -", d_mmmm)

        center_nearest = center_udlr
        d_nearest = d_unlr
        if d_mmlr < d_nearest:
            center_nearest = center_mmlr
            d_nearest = d_mmlr
        if d_udmm < d_nearest:
            center_nearest = center_udmm
            d_nearest = d_udmm
        if d_mmmm < d_nearest:
            center_nearest = center_mmmm
        self.center = center_nearest

        return self.center

    def base_syndrome(self):
        offset = self.center_img - self.center
        b_s = np.zeros(self.syndrome.shape, dtype=np.int8)
        new_xs = (self.xs + offset[0]) % (2 * self.d)
        new_ys = (self.ys + offset[1]) % (2 * self.d)
        b_s[new_xs, new_ys] = 1
        return b_s

    def base_syndrome_xs_ys(self):
        offset = self.center_img - self.center
        b_s = np.zeros(self.syndrome.shape, dtype=np.int8)
        new_xs = (self.xs + offset[0]) % (2 * self.d)
        new_ys = (self.ys + offset[1]) % (2 * self.d)
        b_s[new_xs, new_ys] = 1
        return b_s, new_xs, new_ys



    """
    axis ->
        0: 垂直轴
        1: 水平轴
        2: 左 45 度
        3: 右 45 度
    """
    def reflection_syndrome(self, axis):
        # log(f"self.center = {self.center}")
        r_s = np.zeros(self.syndrome.shape, dtype=np.int8)     # base_syndrome
        if axis == 0:
            axis_symmetry_y = self.center[1]
            for i in range(self.xs.shape[0]):
                x, y = self.xs[i], self.ys[i]
                y = (2 * axis_symmetry_y - y) % (2 * self.d)    # 周期性边界下的垂直轴对称
                r_s[x, y] = 1
        elif axis == 1:
            axis_symmetry_x = self.center[0]
            for i in range(self.xs.shape[0]):
                x, y = self.xs[i], self.ys[i]
                x = (2 * axis_symmetry_x - x) % (2 * self.d)
                r_s[x, y] = 1
        elif axis == 2:
            """
            set: center = (cx, cy)
            k = (cx - cy) / 2
            give (a, b) -> (cy + k, cx - k)[mod 2d]
            核心思量：沿着中心对称轴距离主对角线的距离先移动一段距离 |k|，坐标交换后返回
            """
            offset_center2xey_sub_times2 = self.center[0] - self.center[1]
            # log(f"offset_center2xey_sub_times2 = {offset_center2xey_sub_times2}")
            for i in range(self.xs.shape[0]):
                x, y = self.xs[i], self.ys[i]
                r_s[(y + offset_center2xey_sub_times2) % (2 * self.d), (2 * self.d + x - offset_center2xey_sub_times2) % (2 * self.d)] = 1
                # log(f"({x}, {y})->({y + offset_center2xey_sub_times2}, {x - offset_center2xey_sub_times2})")
        elif axis == 3:
            """
            set: center = (cx, cy)
            k = (cx + cy - (2d-1)) / 2
            give (a, b) -> ((2d-1) - b + 2k), (2d-1) - a + 2k)
            核心思想：沿着中心对称轴距离副对角线的距离先移动一段距离 |k|，坐标交换后用 2d-1 减，之后返回去
            """
            # log(f"self.center = {self.center}")
            offset_center2xpyed_sub_times2 = self.center[0] + self.center[1] - (2 * self.d - 1)
            # log(f"offset_center2xey_sub_times2 = {offset_center2xpyed_sub_times2}")
            for i in range(self.xs.shape[0]):
                x, y = self.xs[i], self.ys[i]
                r_s[(2 * self.d - 1 - y + offset_center2xpyed_sub_times2) % (2 * self.d), (2 * self.d - 1 - x + offset_center2xpyed_sub_times2) % (2 * self.d)] = 1
                # log(f"({x}, {y})->({(2 * self.d - 1 - y + offset_center2xpyed_sub_times2) % (2 * self.d)}, {(2 * self.d - 1 - x + offset_center2xpyed_sub_times2) % (2 * self.d)})")
        else:
            log("Illigal axis")
        return r_s

    def rotation_syndrome(self, theta):
        r_s = np.zeros(self.syndrome.shape, dtype=np.int8)
        if theta == 0:
            for i in range(self.xs.shape[0]):
                x, y = self.xs[i], self.ys[i]
                r_s[(2 * self.d + y + self.center[0] - self.center[1]) % (2 * self.d), (2 * self.d - x + self.center[0] + self.center[1]) % (2 * self.d)] = 1
        elif theta == 1:
            for i in range(self.xs.shape[0]):
                x, y = self.xs[i], self.ys[i]
                r_s[(2 * self.center[0] - x) % (2 * self.d), (2 * self.center[1] - y) % (2 * self.d)] = 1
        elif theta == 2:
            for i in range(self.xs.shape[0]):
                x, y = self.xs[i], self.ys[i]
                r_s[(2 * self.d + self.center[0] + self.center[1] - y) % (2 * self.d), (2 * self.d + self.center[1] - self.center[0] + x) % (2 * self.d)] = 1
        return r_s

    def __eq__(self, other):
        if isinstance(other, SymmetricSyndrome):
            # log("here")
            result = True
            if 'tl' in self.s_types:
                result = result and np.array_equal(self.base_syndrome(), other.base_syndrome())
            else:
                result = result and np.array_equal(self.syndrome, other.syndrome)
            return result
        return NotImplemented

    def __hash__(self):
        # log("there")
        if 'tl' in self.s_types:
            # log("there: tl")
            bs = self.base_syndrome()
            # log("bs: {}".format(bs))
            numpy_bytes = self.base_syndrome().tobytes()
        else:
            # log("there: not tl")
            numpy_bytes = self.syndrome.tobytes()
        return int(hashlib.sha256(numpy_bytes).hexdigest(), 16)


if __name__ == '__main__':
    d = 3
    s_types = {'rf:0'}
    # syndrome = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    syndrome = np.array([0, 1, 0,
                         0, 0, 1,
                         0, 0, 0,
                         0, 0, 1,
                         0, 0, 1,
                         0, 0, 0])
    syndrome02 = np.array([0, 0, 0,
                           1, 0, 0,
                           1, 0, 0,
                           0, 1, 0,
                           0, 0, 0,
                           1, 0, 0])
    syndrome02 = syndrome02[1:-1]
    syndrome = syndrome[1:-1]
    # syndrome = np.array([1, 0, 0, 0, 0,
    #                      0, 0, 0, 0, 0,
    #                      0, 0, 0, 0, 0,
    #                      0, 0, 0, 0, 0,
    #                      0, 0, 0, 0, 1,
    #                      1, 0, 0, 0, 0,
    #                      0, 0, 0, 0, 0,
    #                      0, 0, 0, 0, 0,
    #                      0, 0, 0, 0, 0,
    #                      1, 0, 0, 0, 0])
    symmetric_syndrome = SymmetricSyndrome(d, syndrome, s_types)
    log(symmetric_syndrome.syndrome)
    draw_toric_code(d, symmetric_syndrome, center=True)
    print(symmetric_syndrome.center)
    symmetric_syndrome.syndrome = symmetric_syndrome.reflection_syndrome(2)
    log(symmetric_syndrome.syndrome)
    # symmetric_syndrome.center = (symmetric_syndrome.center_img[0], symmetric_syndrome.center_img[1])
    draw_toric_code(d, symmetric_syndrome, center=True)
