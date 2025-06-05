import numpy as np
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

    calculator = HeatTransferCalculator()
    # 从初始温度场获取网格尺寸
    if not isinstance(initial_temp, np.ndarray):
        initial_temp = np.array(initial_temp)
    nx, ny = initial_temp.shape

    # 网格生成
    dx = Lx / (nx)
    dy = Ly / (ny)
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
    T_old = T.copy()

    # 时间步进
    n_steps = int(total_time / dt)
    time_history = []
    temp_history = []

    for n in range(n_steps):
        T_old = T.copy()

        # 边界条件处理
        # 底部 (对称边界)
        T[0, :] = T[1, :]
        # 左侧 (对称边界)
        T[:, 0] = T[:, 1]

        # 内部节点 - 显式离散
        for i in range(1, nx - 2):
            for j in range(1, ny - 2):
                # 获取当前温度下的物性参数
                k = get_conductivity(T_old[j, i])
                rho = get_density(T_old[j, i])
                cp = get_specific_heat(T_old[j, i])
                alpha = k / (rho * cp)  # 热扩散系数

                # 显式格式温度更新
                T[j, i] = T_old[j, i] + alpha * dt * (
                    (T_old[j, i + 1] - 2 * T_old[j, i] + T_old[j, i - 1])
                    / dx**2  # x方向二阶导数
                    + (T_old[j + 1, i] - 2 * T_old[j, i] + T_old[j - 1, i])
                    / dy**2  # y方向二阶导数
                )

        # 顶部边界处理
        j = ny - 1
        if boundary_type == 3:
            for i in range(1, nx - 2):
                k = get_conductivity(T[j, i])
                q_rad = calculator.air_cooling_heat_flux(
                    T_s=T_old[i, j], T_a=T_inf_top, emissivity=sigma
                )
                q_total = h_top * (T_old[j, i] - T_inf_top) + q_rad
                rho = get_density(T_old[i, j])
                cp = get_specific_heat(T_old[i, j])
                alpha = k / (rho * cp)
                T[i, j] = T_old[i, j] + alpha * dt * (
                    (T_old[i + 1, j] - 2 * T_old[i, j] + T_old[i - 1, j]) / dx**2
                    + (T_old[i, j - 1] - T_old[i, j]) / dy**2
                    - q_total / (k * dy)
                )
        else:
            for i in range(1, nx - 2):
                k = get_conductivity(T_old[j, i])
                rho = get_density(T_old[j, i])
                cp = get_specific_heat(T_old[j, i])
                alpha = k / (rho * cp)
                T[i, j] = T_old[i, j] + alpha * dt * (
                    (T_old[i + 1, j] - 2 * T_old[i, j] + T_old[i - 1, j]) / dx**2
                    + (T_old[i, j - 1] - T_old[i, j]) / dy**2
                    - (calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B)) / (k * dy)
                )

        # 右侧边界处理
        i = nx - 1
        if boundary_type == 3:
            for j in range(1, ny - 2):
                k = get_conductivity(T[i, j])
                q_rad = calculator.air_cooling_heat_flux(
                    T_s=T_old[i, j], T_a=T_inf_right, emissivity=sigma
                )
                q_total = h_right * (T_old[i, j] - T_inf_right) + q_rad
                rho = get_density(T_old[i, j])
                cp = get_specific_heat(T_old[i, j])
                alpha = k / (rho * cp)
                T[i, j] = T_old[i, j] + alpha * dt * (
                    (T_old[i - 1, j] - T_old[i, j]) / dx**2
                    + (T_old[i, j + 1] - 2 * T_old[i, j] + T_old[i, j - 1]) / dy**2
                    - q_total / (k * dx)
                )
        else:
            for j in range(1, ny - 2):
                k = get_conductivity(T_old[i, j])
                rho = get_density(T_old[i, j])
                cp = get_specific_heat(T_old[i, j])
                alpha = k / (rho * cp)
                T[i, j] = T_old[i, j] + alpha * dt * (
                    (T_old[i - 1, j] - T_old[i, j]) / dx**2
                    + (T_old[i, j + 1] - 2 * T_old[i, j] + T_old[i, j - 1]) / dy**2
                    + (calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B)) / (k * dx)
                )

        # 角点处理
        i, j = nx - 1, ny - 1
        k = get_conductivity(T[ny - 1, nx - 1])
        if boundary_type == 3:
            q_rad_top = calculator.air_cooling_heat_flux(
                T_s=T_old[i, j], T_a=T_inf_top, emissivity=sigma
            )
            q_rad_right = calculator.air_cooling_heat_flux(
                T_s=T_old[i, j], T_a=T_inf_right, emissivity=sigma
            )
            q_total_top = h_top * (T_old[i, j] - T_inf_top) + q_rad_top
            q_total_right = h_right * (T_old[i, j] - T_inf_right) + q_rad_right
            T[i, j] = T_old[i, j] + alpha * dt * (
                (T_old[i - 1, j] - T_old[i, j]) / dx**2
                + (T_old[i, j - 1] - T_old[i, j]) / dy**2
                - q_total_top / (k * dy)
                - q_total_right / (k * dx)
            )
        else:
            k = get_conductivity(T_old[nx - 1, ny - 1])
            rho = get_density(T_old[nx - 1, ny - 1])
            cp = get_specific_heat(T_old[nx - 1, ny - 1])
            alpha = k / (rho * cp)
            T[i, j] = T_old[i, j] + alpha * dt * (
                (T_old[i - 1, j] - T_old[i, j]) / dx**2
                + (T_old[i, j - 1] - T_old[i, j]) / dy**2
                - (calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B)) / (k * dy)
                - (calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B)) / (k * dx)
            )

        # 记录时间和温度（可选）
        if n % 100 == 0:
            time_history.append(n * dt)
            temp_history.append(T.copy())

        # 收敛检测
        # 每次迭代都检查收敛状态
        converged = np.max(np.abs(T - T_old)) < tol
        if converged:  # 达到收敛条件时输出
            print(f"\n在 t = {n*dt:.2f}s 达到收敛条件")
            # print(f"当前温度范围: {np.min(T):.1f}℃ ~ {np.max(T):.1f}℃")

    return X, Y, T, time_history, temp_history


def solve_transient_heat_conduction_with_phase_properties(
    Lx,
    Ly,
    T_inf_top,
    T_inf_right,
    dt,
    total_time,
    initial_temp,
    Ts,  # 固相线温度 [K]
    Tl,  # 液相线温度 [K]
    props,  # 钢种热物性字典
    boundary_type=3,
    q_k_A=2860,
    q_k_B=276,
    h_top=0.0,
    h_right=0.0,
    sigma=0.8,
    tol=1e-5,
):
    """
    使用相态相关物性计算的二维瞬态热传导求解器

    参数:
        Lx, Ly: 区域长度和宽度 [m]
        T_inf_top, T_inf_right: 环境温度 [℃]
        dt: 时间步长 [s]
        total_time: 总模拟时间 [s]
        initial_temp: 二维初始温度场 [K]
        Ts, Tl: 固相线和液相线温度 [K]
        props: 钢种热物性参数字典
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
    from boundary_condition import HeatTransferCalculator

    calculator = HeatTransferCalculator()

    # 从初始温度场获取网格尺寸
    if not isinstance(initial_temp, np.ndarray):
        initial_temp = np.array(initial_temp)
    nx, ny = initial_temp.shape
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # 初始热扩散率检查（使用初始温度场的平均温度）
    avg_temp = np.mean(initial_temp)
    k_init = lamda_cal(avg_temp, 0, Ts, Tl, 0, props)
    rho_init = rho_cal(avg_temp, Ts, Tl, props)
    cp_init = cp_cal(avg_temp, Ts, Tl, props)
    alpha_init = k_init / (rho_init * cp_init)
    Fo_x = alpha_init * dt / dx**2
    Fo_y = alpha_init * dt / dy**2
    if Fo_x > 0.5 or Fo_y > 0.5:
        raise ValueError(
            f"时间步长过大，需满足 Fourier数 <= 0.5 (当前 Fo_x={Fo_x:.2f}, Fo_y={Fo_y:.2f})"
        )

    # 初始化温度场
    T = initial_temp.copy()
    T_old = T.copy()

    # 时间步进
    n_steps = int(total_time / dt)
    time_history = []
    temp_history = []

    for n in range(n_steps):
        T_old = T.copy()

        # 边界条件处理
        T[0, :] = T[1, :]
        T[:, 0] = T[:, 1]

        # 内部节点 - 显式离散
        for i in range(1, nx - 2):
            for j in range(1, ny - 2):
                k = lamda_cal(T_old[i, j], (i + j) / 2, Ts, Tl, 0, props)
                rho = rho_cal(T_old[i, j], Ts, Tl, props)
                cp = cp_cal(T_old[i, j], Ts, Tl, props)
                alpha = k / (rho * cp)
                T[i, j] = T_old[i, j] + alpha * dt * (
                    (T_old[i + 1, j] - 2 * T_old[i, j] + T_old[i - 1, j]) / dx**2
                    + (T_old[i, j + 1] - 2 * T_old[i, j] + T_old[i, j - 1]) / dy**2
                )

        # 顶部边界处理
        j = ny - 1
        if boundary_type == 3:
            for i in range(1, nx - 2):
                k = lamda_cal(T[i, j], (i + j) / 2, Ts, Tl, 0, props)
                q_rad = calculator.air_cooling_heat_flux(
                    T_s=T_old[i, j], T_a=T_inf_top, emissivity=sigma
                )
                q_total = h_top * (T_old[i, j] - T_inf_top) + q_rad
                rho = rho_cal(T_old[i, j], Ts, Tl, props)
                cp = cp_cal(T_old[i, j], Ts, Tl, props)
                alpha = k / (rho * cp)
                T[i, j] = T_old[i, j] + alpha * dt * (
                    (T_old[i + 1, j] - 2 * T_old[i, j] + T_old[i - 1, j]) / dx**2
                    + (T_old[i, j - 1] - T_old[i, j]) / dy**2
                    - q_total / (k * dy)
                )
        else:
            for i in range(1, nx - 2):
                k = lamda_cal(T_old[i, j], (i + j) / 2, Ts, Tl, 0, props)
                rho = rho_cal(T_old[i, j], Ts, Tl, props)
                cp = cp_cal(T_old[i, j], Ts, Tl, props)
                alpha = k / (rho * cp)
                T[i, j] = T_old[i, j] + alpha * dt * (
                    (T_old[i + 1, j] - 2 * T_old[i, j] + T_old[i - 1, j]) / dx**2
                    + (T_old[i, j - 1] - T_old[i, j]) / dy**2
                    + calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B) / (k * dx)
                )

        # 右侧边界处理
        i = nx - 1
        if boundary_type == 3:
            for j in range(1, ny - 2):
                k = lamda_cal(T[i, j], (i + j) / 2, Ts, Tl, 0, props)
                q_rad = calculator.air_cooling_heat_flux(
                    T_s=T_old[i, j], T_a=T_inf_right, emissivity=sigma
                )
                q_total = h_right * (T_old[i, j] - T_inf_right) + q_rad
                rho = rho_cal(T_old[i, j], Ts, Tl, props)
                cp = cp_cal(T_old[i, j], Ts, Tl, props)
                alpha = k / (rho * cp)
                T[i, j] = T_old[i, j] + alpha * dt * (
                    (T_old[i - 1, j] - T_old[i, j]) / dx**2
                    + (T_old[i, j + 1] - 2 * T_old[i, j] + T_old[i, j - 1]) / dy**2
                    - q_total / (k * dx)
                )
        else:
            for j in range(1, ny - 2):
                k = lamda_cal(T_old[i, j], (i + j) / 2, Ts, Tl, 0, props)
                rho = rho_cal(T_old[i, j], Ts, Tl, props)
                cp = cp_cal(T_old[i, j], Ts, Tl, props)
                alpha = k / (rho * cp)
                T[i, j] = T_old[i, j] + alpha * dt * (
                    (T_old[i - 1, j] - T_old[i, j]) / dx**2
                    + (T_old[i, j + 1] - 2 * T_old[i, j] + T_old[i, j - 1]) / dy**2
                    + calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B) / (k * dx)
                )

        # 角点处理
        i, j = nx - 1, ny - 1
        k = lamda_cal(T[i, j], (i + j) / 2, Ts, Tl, 0, props)
        if boundary_type == 3:
            q_rad_top = calculator.air_cooling_heat_flux(
                T_s=T_old[i, j], T_a=T_inf_top, emissivity=sigma
            )
            q_rad_right = calculator.air_cooling_heat_flux(
                T_s=T_old[i, j], T_a=T_inf_right, emissivity=sigma
            )
            q_total_top = h_top * (T_old[i, j] - T_inf_top) + q_rad_top
            q_total_right = h_right * (T_old[i, j] - T_inf_right) + q_rad_right
            T[i, j] = T_old[i, j] + alpha * dt * (
                (T_old[i - 1, j] - T_old[i, j]) / dx**2
                + (T_old[i, j - 1] - T_old[i, j]) / dy**2
                - q_total_top / (k * dy)
                - q_total_right / (k * dx)
            )
        else:
            k = lamda_cal(T_old[i, j], (i + j) / 2, Ts, Tl, 0, props)
            rho = rho_cal(T_old[i, j], Ts, Tl, props)
            cp = cp_cal(T_old[i, j], Ts, Tl, props)
            alpha = k / (rho * cp)
            T[i, j] = T_old[i, j] + alpha * dt * (
                (T_old[i - 1, j] - T_old[i, j]) / dx**2
                + (T_old[i, j - 1] - T_old[i, j]) / dy**2
                - calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B) / (k * dy)
                - calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B) / (k * dx)
            )

        # 记录时间和温度（可选）
        if n % 100 == 0:
            time_history.append(n * dt)
            temp_history.append(T.copy())

        # 稳态检测
        if np.max(np.abs(T - T_old)) < tol:
            print(f"\n稳态在 t = {n*dt:.2f}s 达到")
            print(f"最终温度范围: {np.min(T):.1f}℃ ~ {np.max(T):.1f}℃")
            break

    return X, Y, T, time_history, temp_history


if __name__ == "__main__":
    # 测试基础热传导求解器
    print("\n=== 测试基础热传导求解器 ===")
    Lx, Ly = 0.15, 0.15  # 区域尺寸(m)
    dt = 0.1  # 时间步长(s)
    total_time = 10  # 总时间(s)
    initial_temp = np.full((20, 20), 1500)  # 初始温度场(10x10网格,1500℃)

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
        h_top=10000,
        h_right=5000,
        sigma=0.8,
    )
    print(f"最终温度范围: {np.min(T):.1f}℃ ~ {np.max(T):.1f}℃")

    # 测试第二类边界条件
    print("\n测试第二类边界条件:")
    X, Y, T, times, temps = solve_transient_heat_conduction(
        Lx,
        Ly,
        20,
        20,
        dt,
        total_time,
        initial_temp,
        boundary_type=2,
        q_k_A=2860,
        q_k_B=276,
    )
    print(f"最终温度范围: {np.min(T):.1f}℃ ~ {np.max(T):.1f}℃")
