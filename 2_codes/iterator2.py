import numpy as np
from icecream import ic
from prop_vs_temp import (
    cp_cal,
    get_conductivity,
    get_density,
    get_specific_heat,
    lamda_cal,
    rho_cal,
)


def solve_transient_heat_conduction(
    Lx,
    Ly,
    T_inf_top,
    T_inf_right,
    dt,
    total_time,
    initial_temp,  # 二维初始温度场 [℃]
    boundary_type=3,  # 2为第二类边界条件，3为第三类边界条件
    q_k_A=2860,  # 顶部热流密度系数A [kW/m²] (第二类边界)
    q_k_B=276,  # 右侧热流密度系数B [kW/m²] (第二类边界)
    h_top=0.0,  # 顶部为第三类边界条件
    h_right=0.0,  # 右侧为第三类边界条件
    sigma=0.8,  # 辐射率 (第三类边界)
    tol=1e-5,  # 使用全局定义的容差
):
    """
    二维瞬态热传导问题求解（显式格式）

    参数:
        Lx, Ly: 区域长度和宽度 [m]
        h_top, h_right: 对流换热系数 [W/(m²·K)]
        T_inf_top, T_inf_right: 环境温度 [℃]
        dt: 时间步长 [s]
        total_time: 总模拟时间 [s]
        initial_temp: 二维初始温度场 [K] (数组shape决定网格数)
        boundary_type: 边界条件类型 (2为第二类，3为第三类)
        q_k_A, q_k_B: 第二类边界条件的热流密度系数
        h_top, h_right: 第三类边界条件的对流换热系数
        sigma: 辐射率 (第三类边界)
        tol: 收敛容差（用于稳态检测）

    返回:
        X, Y: 网格坐标
        T: 最终温度场
        time_history: 时间历史记录
        temp_history: 温度历史记录
    """
    import numpy as np
    from boundary_condition import HeatTransferCalculator

    # avg_temp = np.mean(initial_temp)
    # k_init = get_conductivity(avg_temp)
    # rho_init = get_density(avg_temp)
    # cp_init = get_specific_heat(avg_temp)
    # alpha_init = k_init / (rho_init * cp_init)
    # Fo_x = alpha_init * dt / dx**2
    # Fo_y = alpha_init * dt / dy**2
    # if Fo_x > 0.5 or Fo_y > 0.5:
    #     raise ValueError(
    #         f"时间步长过大，需满足 Fourier数 <= 0.5 (当前 Fo_x={Fo_x:.2f}, Fo_y={Fo_y:.2f})"
    #     )

    calculator = HeatTransferCalculator()
    # 从初始温度场获取网格尺寸
    if not isinstance(initial_temp, np.ndarray):
        initial_temp = np.array(initial_temp)
    nx, ny = initial_temp.shape

    nx += 1
    ny += 1
    # 网格生成
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    # print(f"网格尺寸: {nx} x {ny}")
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # 初始热扩散率检查（使用初始温度场的平均温度）
    avg_temp = np.mean(initial_temp)
    k_init = get_conductivity(avg_temp)
    rho_init = get_density(avg_temp)
    cp_init = get_specific_heat(avg_temp)
    alpha_init = k_init / (rho_init * cp_init)
    Fo_x = alpha_init * dt / dx**2
    Fo_y = alpha_init * dt / dy**2
    if Fo_x > 0.5 or Fo_y > 0.5:
        raise ValueError(
            f"时间步长过大，需满足 Fourier数 <= 0.5 (当前 Fo_x={Fo_x:.2f}, Fo_y={Fo_y:.2f})"
        )

    # 初始化温度场
    T = initial_temp.copy()

    # 时间步进
    n_steps = int(total_time / dt)
    time_history = []
    temp_history = []

    """ 数值计算 """
    eps = tol
    iterStep = n_steps

    nt = int(total_time / dt)

    for i in range(nt):
        T_old_iter = T.copy()
        for j in range(iterStep):
            Tn = T.copy()

            # 计算全场物性参数 (向量化操作)
            k = get_conductivity(T_old_iter)
            rho = get_density(T_old_iter)
            cp = get_specific_heat(T_old_iter)
            alpha = k / (rho * cp)  # 热扩散系数数组

            # 中心节点
            T[1:-1, 1:-1] = (
                T_old_iter[1:-1, 1:-1]
                + (
                    dt
                    / (dx * dx)
                    * (
                        T[1:-1, 2:] * alpha[1:-1, 2:]
                        - 2 * T[1:-1, 1:-1] * alpha[1:-1, 1:-1]
                        + T[1:-1, 0:-2] * alpha[1:-1, 0:-2]
                    )
                )
                + (
                    dt
                    / (dy * dy)
                    * (
                        T[2:, 1:-1] * alpha[2:, 1:-1]
                        - 2 * T[1:-1, 1:-1] * alpha[1:-1, 1:-1]
                        + T[0:-2, 1:-1] * alpha[0:-2, 1:-1]
                    )
                )
            )
            if boundary_type == 3:

                # 上边界-边
                T[-1, 1:-1] = T_old_iter[-1, 1:-1] + (
                    (
                        -dx
                        * (
                            calculator.air_cooling_heat_flux(
                                T_s=T[-1, 1:-1], T_a=T_inf_top, emissivity=sigma
                            )
                            + h_top * (T[-1, 1:-1] - T_inf_top)
                        )
                    )
                    - k[-1, 1:-1] * (T[-1, 1:-1] - T[-1, 2:]) * dy / (2 * dx)
                    - k[-1, 1:-1] * (T[-1, 1:-1] - T[-1, 0:-2]) * dy / (2 * dx)
                    - k[-1, 1:-1] * (T[-1, 1:-1] - T[-2, 1:-1]) * dx / dy
                ) * dt / (rho[-1, 1:-1] * cp[-1, 1:-1] * 0.5 * dx * dy)

                # 下边界-边
                T[0, 1:-1] = T_old_iter[0, 1:-1] + (
                    (
                        -dx
                        * (
                            calculator.air_cooling_heat_flux(
                                T_s=T[0, 1:-1], T_a=T_inf_top, emissivity=sigma
                            )
                            + h_top * (T[0, 1:-1] - T_inf_top)
                        )
                    )
                    - (k[0, 1:-1] * (T[0, 1:-1] - T[0, 2:]) * dy / (2 * dx))
                    - (k[0, 1:-1] * (T[0, 1:-1] - T[0, 0:-2]) * dy / (2 * dx))
                    - (k[0, 1:-1] * (T[0, 1:-1] - T[1, 1:-1]) * dx / dy)
                ) * dt / (rho[0, 1:-1] * cp[0, 1:-1] * 0.5 * dx * dy)

                # 左边界-边
                T[1:-1, 0] = T_old_iter[1:-1, 0] + (
                    (
                        -dy
                        * (
                            calculator.air_cooling_heat_flux(
                                T_s=T[1:-1, 0], T_a=T_inf_right, emissivity=sigma
                            )
                            + h_right * (T[1:-1, 0] - T_inf_right)
                        )
                    )
                    - (k[1:-1, 0] * (T[1:-1, 0] - T[2:, 0]) * dx / (2 * dy))
                    - (k[1:-1, 0] * (T[1:-1, 0] - T[0:-2, 0]) * dx / (2 * dy))
                    - (k[1:-1, 0] * (T[1:-1, 0] - T[1:-1, 1]) * dy / dx)
                ) * dt / (rho[1:-1, 0] * cp[1:-1, 0] * 0.5 * dx * dy)

                # 右边界-边
                T[1:-1, -1] = T_old_iter[1:-1, -1] + (
                    (
                        -dy
                        * (
                            calculator.air_cooling_heat_flux(
                                T_s=T[1:-1, -1], T_a=T_inf_right, emissivity=sigma
                            )
                            + h_right * (T[1:-1, -1] - T_inf_right)
                        )
                    )
                    - (k[1:-1, -1] * (T[1:-1, -1] - T[2:, -1]) * dx / (2 * dy))
                    - (k[1:-1, -1] * (T[1:-1, -1] - T[0:-2, -1]) * dx / (2 * dy))
                    - (k[1:-1, -1] * (T[1:-1, -1] - T[1:-1, -2]) * dy / dx)
                ) * dt / (rho[1:-1, -1] * cp[1:-1, -1] * 0.5 * dx * dy)

                # 左上边界-点
                T[-1, 0] = T_old_iter[-1, 0] + (
                    (
                        -(0.5 * dy)
                        * (
                            calculator.air_cooling_heat_flux(
                                T_s=T[-1, 0], T_a=T_inf_right, emissivity=sigma
                            )
                            + h_right * (T[-1, 0] - T_inf_right)
                        )
                    )
                    + (
                        -(0.5 * dx)
                        * (
                            calculator.air_cooling_heat_flux(
                                T_s=T[-1, 0], T_a=T_inf_top, emissivity=sigma
                            )
                            + h_top * (T[-1, 0] - T_inf_top)
                        )
                    )
                    - k[-1, 0] * (T[-1, 0] - T[-2, 0]) * dx / (2 * dy)
                    - k[-1, 0] * (T[-1, 0] - T[-1, 1]) * dy / (2 * dx)
                ) * dt / (rho[-1, 0] * cp[-1, 0] * 0.5 * dx * 0.5 * dy)
                ic(T[-1, 0])
                # 左下边界-点
                T[0, 0] = T_old_iter[0, 0] + (
                    (
                        -0.5
                        * dy
                        * (
                            calculator.air_cooling_heat_flux(
                                T_s=T[0, 0], T_a=T_inf_right, emissivity=sigma
                            )
                            + h_right * (T[0, 0] - T_inf_right)
                        )
                    )
                    + (
                        -0.5
                        * dx
                        * (
                            calculator.air_cooling_heat_flux(
                                T_s=T[0, 0], T_a=T_inf_top, emissivity=sigma
                            )
                            + h_top * (T[0, 0] - T_inf_top)
                        )
                    )
                    - k[0, 0] * (T[0, 0] - T[1, 0]) * dx / (2 * dy)
                    - k[0, 0] * (T[0, 0] - T[0, 1]) * dy / (2 * dx)
                ) * dt / (rho[0, 0] * cp[0, 0] * 0.5 * dx * 0.5 * dy)

                # 右上边界-点
                T[-1, -1] = T_old_iter[-1, -1] + (
                    (
                        -0.5
                        * dy
                        * (
                            calculator.air_cooling_heat_flux(
                                T_s=T[-1, -1], T_a=T_inf_right, emissivity=sigma
                            )
                            + h_right * (T[-1, -1] - T_inf_right)
                        )
                    )
                    + (
                        -0.5
                        * dx
                        * (
                            calculator.air_cooling_heat_flux(
                                T_s=T[-1, -1], T_a=T_inf_top, emissivity=sigma
                            )
                            + h_top * (T[-1, -1] - T_inf_top)
                        )
                    )
                    - k[-1, -1] * (T[-1, -1] - T[-2, -1]) * dx / (2 * dy)
                    - k[-1, -1] * (T[-1, -1] - T[-1, -2]) * dy / (2 * dx)
                ) * dt / (rho[-1, -1] * cp[-1, -1] * 0.5 * dx * 0.5 * dy)

                # 右下边界-点
                T[0, -1] = T_old_iter[0, -1] + (
                    (
                        -(0.5 * dy)
                        * (
                            calculator.air_cooling_heat_flux(
                                T_s=T[0, -1], T_a=T_inf_right, emissivity=sigma
                            )
                            + h_right * (T[0, -1] - T_inf_right)
                        )
                    )
                    + (
                        -(0.5 * dx)
                        * (
                            calculator.air_cooling_heat_flux(
                                T_s=T[0, -1], T_a=T_inf_top, emissivity=sigma
                            )
                            + h_top * (T[0, -1] - T_inf_top)
                        )
                    )
                    - k[0, -1] * (T[0, -1] - T[1, -1]) * dx / (2 * dy)
                    - k[0, -1] * (T[0, -1] - T[0, -2]) * dy / (2 * dx)
                ) * dt / (rho[0, -1] * cp[0, -1] * 0.5 * dx * 0.5 * dy)
            else:
                # 上边界-边
                T[-1, 1:-1] = T_old_iter[-1, 1:-1] + (
                    (-dx * calculator.mold_heat_flux(i * dt, A=q_k_A, B=q_k_B))
                    - k[-1, 1:-1] * (T[-1, 1:-1] - T[-1, 2:]) * dy / (2 * dx)
                    - k[-1, 1:-1] * (T[-1, 1:-1] - T[-1, 0:-2]) * dy / (2 * dx)
                    - k[-1, 1:-1] * (T[-1, 1:-1] - T[-2, 1:-1]) * dx / dy
                ) * dt / (rho[-1, 1:-1] * cp[-1, 1:-1] * 0.5 * dx * dy)

                # 下边界-边
                T[0, 1:-1] = T_old_iter[0, 1:-1] + (
                    (-dx * calculator.mold_heat_flux(i * dt, A=q_k_A, B=q_k_B))
                    - (k[0, 1:-1] * (T[0, 1:-1] - T[0, 2:]) * dy / (2 * dx))
                    - (k[0, 1:-1] * (T[0, 1:-1] - T[0, 0:-2]) * dy / (2 * dx))
                    - (k[0, 1:-1] * (T[0, 1:-1] - T[1, 1:-1]) * dx / dy)
                ) * dt / (rho[0, 1:-1] * cp[0, 1:-1] * 0.5 * dx * dy)

                # 左边界-边
                T[1:-1, 0] = T_old_iter[1:-1, 0] + (
                    (-dy * calculator.mold_heat_flux(i * dt, A=q_k_A, B=q_k_B))
                    - (k[1:-1, 0] * (T[1:-1, 0] - T[2:, 0]) * dx / (2 * dy))
                    - (k[1:-1, 0] * (T[1:-1, 0] - T[0:-2, 0]) * dx / (2 * dy))
                    - (k[1:-1, 0] * (T[1:-1, 0] - T[1:-1, 1]) * dy / dx)
                ) * dt / (rho[1:-1, 0] * cp[1:-1, 0] * 0.5 * dx * dy)

                # 右边界-边
                T[1:-1, -1] = T_old_iter[1:-1, -1] + (
                    (-dy * calculator.mold_heat_flux(i * dt, A=q_k_A, B=q_k_B))
                    - (k[1:-1, -1] * (T[1:-1, -1] - T[2:, -1]) * dx / (2 * dy))
                    - (k[1:-1, -1] * (T[1:-1, -1] - T[0:-2, -1]) * dx / (2 * dy))
                    - (k[1:-1, -1] * (T[1:-1, -1] - T[1:-1, -2]) * dy / dx)
                ) * dt / (rho[1:-1, -1] * cp[1:-1, -1] * 0.5 * dx * dy)

                # 左上边界-点
                T[-1, 0] = T_old_iter[-1, 0] + (
                    (-(0.5 * dy) * calculator.mold_heat_flux(i * dt, A=q_k_A, B=q_k_B))
                    + (
                        -(0.5 * dx)
                        * calculator.mold_heat_flux(i * dt, A=q_k_A, B=q_k_B)
                    )
                    - k[-1, 0] * (T[-1, 0] - T[-2, 0]) * dx / (2 * dy)
                    - k[-1, 0] * (T[-1, 0] - T[-1, 1]) * dy / (2 * dx)
                ) * dt / (rho[-1, 0] * cp[-1, 0] * 0.5 * dx * 0.5 * dy)

                # 左下边界-点
                T[0, 0] = T_old_iter[0, 0] + (
                    (-0.5 * dy * calculator.mold_heat_flux(i * dt, A=q_k_A, B=q_k_B))
                    + (-0.5 * dx * calculator.mold_heat_flux(i * dt, A=q_k_A, B=q_k_B))
                    - k[0, 0] * (T[0, 0] - T[1, 0]) * dx / (2 * dy)
                    - k[0, 0] * (T[0, 0] - T[0, 1]) * dy / (2 * dx)
                ) * dt / (rho[0, 0] * cp[0, 0] * 0.5 * dx * 0.5 * dy)

                # 右上边界-点
                T[-1, -1] = T_old_iter[-1, -1] + (
                    (-0.5 * dy * calculator.mold_heat_flux(i * dt, A=q_k_A, B=q_k_B))
                    + (-0.5 * dx * calculator.mold_heat_flux(i * dt, A=q_k_A, B=q_k_B))
                    - k[-1, -1] * (T[-1, -1] - T[-2, -1]) * dx / (2 * dy)
                    - k[-1, -1] * (T[-1, -1] - T[-1, -2]) * dy / (2 * dx)
                ) * dt / (rho[-1, -1] * cp[-1, -1] * 0.5 * dx * 0.5 * dy)

                # 右下边界-点
                T[0, -1] = T_old_iter[0, -1] + (
                    (-(0.5 * dy) * calculator.mold_heat_flux(i * dt, A=q_k_A, B=q_k_B))
                    + (
                        -(0.5 * dx)
                        * calculator.mold_heat_flux(i * dt, A=q_k_A, B=q_k_B)
                    )
                    - k[0, -1] * (T[0, -1] - T[1, -1]) * dx / (2 * dy)
                    - k[0, -1] * (T[0, -1] - T[0, -2]) * dy / (2 * dx)
                ) * dt / (rho[0, -1] * cp[0, -1] * 0.5 * dx * 0.5 * dy)

            error = np.max(np.abs(T - Tn))
            if error < eps:
                # print('No.', i + 1, 'Convergence Y, Time = ', (i + 1) * dt, 'iteration = ', j + 1)
                break
            # if j == iterStep - 1:
            #     ic(f"达到最大迭代次数，{error}")

        time_history.append(i * dt)
        temp_history.append(T.copy())
        ic(i)
    return X, Y, T, time_history, temp_history


def plot_heatmap(T, Lx, Ly):
    """使用plotly绘制交互式温度场热图，保持实际物理尺寸比例"""
    import plotly.graph_objects as go
    import numpy as np

    # 获取网格尺寸
    ny, nx = T.shape

    # 创建网格坐标
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # 创建热图
    fig = go.Figure(
        data=go.Heatmap(
            x=x,
            y=y,
            z=T,
            colorscale="Jet",
            zsmooth="best",  # 平滑显示
            colorbar=dict(title="Temperature (℃)"),
        )
    )

    # 设置布局保持比例
    fig.update_layout(
        title="Temperature Distribution",
        xaxis=dict(
            title=f"X direction (Lx={Lx}m",
            scaleanchor="y",  # 保持xy比例
            constrain="domain",  # 限制在绘图区域内
        ),
        yaxis=dict(title=f"Y direction (Ly={Ly}m", constrain="domain"),
        autosize=False,
        width=800,
        height=800 * (Ly / Lx),  # 根据比例调整高度
        margin=dict(l=50, r=50, b=50, t=50),
    )

    # 显示图形
    fig.write_html("heatmap.html")
    fig.show()


if __name__ == "__main__":
    # 测试基础热传导求解器
    print("\n=== 测试基础热传导求解器 ===")
    Lx, Ly = 0.15, 0.1  # 区域尺寸(m)
    dt = 0.1  # 时间步长(s)

    nx = 30
    ny = 20
    dx = Lx / nx
    dy = Ly / ny
    total_time = 5  # 总时间(s)
    initial_temp = np.full((nx, ny), 1550)  # 初始温度场(10x10网格,1500℃)

    # 测试第三类边界条件
    print("\n测试第三类边界条件:")
    X, Y, T, times, temps = solve_transient_heat_conduction(
        Lx,
        Ly,
        20,
        20,
        dt,
        total_time,
        initial_temp,
        boundary_type=3,
        h_top=630,
        h_right=1900,
        sigma=0.8,
    )
    import matplotlib.pyplot as plt

    plot_heatmap(T, Lx, Ly)
