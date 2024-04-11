import numpy as np
from rich.console import Console

console = Console()


def log(info):
    console.print(info)


def ZX2image(d, zxsyndrom):
    zxsyndrom = np.insert(zxsyndrom, 0, 0)
    zxsyndrom = np.append(zxsyndrom, 0)
    # image_syndrome = np.zeros((2 * d, 2 * d))
    image_syndrome = np.zeros((2 * d, 2 * d), dtype=np.int8)
    m_z_stabilizer = d ** 2

    for i in range(m_z_stabilizer):
        image_syndrome[1 + 2 * (i // d), 1 + 2 * (i % d)] = zxsyndrom[i]
    for i in range(m_z_stabilizer, zxsyndrom.shape[0]):
        image_syndrome[2 * ((i - m_z_stabilizer) // d), 2 * ((i - m_z_stabilizer) % d)] = zxsyndrom[i]
    return image_syndrome

def ZX2image_full(d, zxsyndrom):
    image_syndrome = np.zeros((2 * d, 2 * d))
    m_z_stabilizer = d ** 2

    for i in range(m_z_stabilizer):
        image_syndrome[1 + 2 * (i // d), 1 + 2 * (i % d)] = zxsyndrom[i]
    for i in range(m_z_stabilizer, zxsyndrom.shape[0]):
        image_syndrome[2 * ((i - m_z_stabilizer) // d), 2 * ((i - m_z_stabilizer) % d)] = zxsyndrom[i]
    return image_syndrome


def ZX2image_sur(d, zxsyndrome):
    image_syndrome = np.zeros((2 * d - 1, 2 * d - 1))
    m = 2 * d ** 2 - 2 * d
    side = 2 * d - 1

    for i in range(m):
        # log("i = {}".format(i))
        a = i % side
        b = i // side
        if a < d - 1:
            # log(f"({b*2}, {a * 2 + 1})")
            image_syndrome[b * 2, a * 2 + 1] = 1 if int(zxsyndrome[i]) == 0 else -1
            # image_syndrome[b * 2, (a % 2) * 2 + 1] = 1 if int(zxsyndrome[i]) == 0 else -1
        else:
            # log(f"({b * 2 + 1}, {(a - d + 1) * 2})")
            image_syndrome[b * 2 + 1, (a - d + 1) * 2] = 1 if int(zxsyndrome[i]) == 0 else -1
    return image_syndrome

if __name__ == '__main__':
    syndrome = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0])
    log(ZX2image(3, syndrome))      # 测试成功
