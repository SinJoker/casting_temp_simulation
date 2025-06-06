import numpy as np
from prop_vs_temp import get_conductivity, get_density, get_specific_heat
from icecream import ic


def convergence_detection(initial_temp, dt, dx, dy):
    """
    检测收敛性并返回建议的空间步长，确保数值模拟的时间步长和空间步长满足稳定性条件。

    参数:
    - initial_temp: 初始温度场 (numpy数组)
    - dt: 时间步长 (秒)
    - dx: x方向的空间步长 (米)
    - dy: y方向的空间步长 (米)

    返回:
    dict: 包含收敛状态和建议步长的字典，格式为:
        {
            "is_converged": bool,  # 是否收敛
            "Fo_x": float,        # x方向Fourier数
            "Fo_y": float,        # y方向Fourier数
            "suggested_dx": float, # 建议的x方向步长
            "suggested_dy": float, # 建议的y方向步长
            "message": str        # 状态信息
        }

    异常:
    - ValueError: 如果Fourier数大于或等于0.5，表示时间步长过大，需要调整
    """

    # 计算初始温度场的最大温度
    max_temp = np.max(initial_temp)

    # 获取平均温度下的热导率、密度和比热容
    k_init = get_conductivity(max_temp)
    rho_init = get_density(max_temp)
    cp_init = get_specific_heat(max_temp)

    # 计算初始热扩散率
    alpha_init = k_init / (rho_init * cp_init)

    # 计算Fourier数
    Fo_x = alpha_init * dt / dx**2

    # 定义优先选择的步长值
    # preferred_steps = [0.1, 0.2, 0.5, 0.01, 0.02, 0.05, 0.001, 0.002, 0.005]
    fo_range_min = 0.01
    fo_range_max = 0.05
    dx_max = np.sqrt(alpha_init * dt / fo_range_min)
    dx_min = np.sqrt(alpha_init * dt / fo_range_max)

    if fo_range_min <= Fo_x <= fo_range_max:
        message = f"空间步长较为理想"
    elif Fo_x > fo_range_max:
        message = f"空间步长过小\n建议范围: {dx_min*1000:.1f}~{dx_max*1000:.1f}mm"
    else:
        message = f"空间步长过大\n建议范围: {dx_min*1000:.1f}~{dx_max*1000:.1f}mm"

    return message


if __name__ == "__main__":

    result = convergence_detection(1550, 0.1, 0.005, 0.005)
    print(f"结果: {result}")
