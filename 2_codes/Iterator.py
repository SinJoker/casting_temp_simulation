import numpy as np

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
from prop_vs_temp import (
    get_density,
    get_conductivity,
    get_specific_heat,
    lamda_cal,
    cp_cal,
    rho_cal,
)

# import os


def solve_transient_heat_conduction(
    Lx,
    Ly,
    T_inf_top,
    T_inf_right,
    dt,
    total_time,
    initial_temp,  # 二维初始温度场 [K]
    boundary_type=3,  # 2为第二类边界条件，3为第三类边界条件
    q_k_A=2860,  # 顶部热流密度系数A [kW/m²] (第二类边界)
    q_k_B=276,  # 右侧热流密度系数B [kW/m²] (第二类边界)
    h_top=0.0,  # 顶部为第三类边界条件
    h_right=0.0,  # 右侧为第三类边界条件
    sigma=0.8,  # 辐射率 (第三类边界)
    tol=1e-5,  # 使用全局定义的容差
):
    from boundary_condition import HeatTransferCalculator
    import numpy as np

    calculator = HeatTransferCalculator()

    """
    二维瞬态热传导问题求解（显式格式）

    参数:
        Lx, Ly: 区域长度和宽度 [m]
        h_top, h_right: 对流换热系数 [W/(m²·K)]
        T_inf_top, T_inf_right: 环境温度 [K]
        dt: 时间步长 [s]
        total_time: 总模拟时间 [s]
        initial_temp: 二维初始温度场 [K] (数组shape决定网格数)
        tol: 收敛容差（用于稳态检测）
    """
    if not isinstance(initial_temp, np.ndarray):
        initial_temp = np.array(initial_temp)
    # 从初始温度场获取网格尺寸
    nx, ny = initial_temp.shape

    # 网格生成
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
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

    # print(f"开始模拟，总时间步数: {n_steps}，总时间: {total_time}s")
    # print(f"初始温度场: {initial_temp}℃")
    # print(f"环境温度: 顶部={T_inf_top}℃，右侧={T_inf_right}℃")

    for n in range(n_steps):
        T_old = T.copy()

        # 边界条件处理
        # 底部 (对称边界)
        T[0, :] = T[1, :]
        # 左侧 (对称边界)
        T[:, 0] = T[:, 1]

        # 内部节点 - 显式离散
        # 公式推导过程:
        # 1. 二维非稳态热传导方程:
        #    ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)
        # 2. 离散化处理:
        #    a. 时间导数: 前向差分 (显式格式)
        #       ∂T/∂t ≈ (T_new - T_old)/Δt
        #    b. 空间导数: 中心差分
        #       ∂²T/∂x² ≈ (T[i+1] - 2T[i] + T[i-1])/Δx²
        #       ∂²T/∂y² ≈ (T[j+1] - 2T[j] + T[j-1])/Δy²
        # 3. 离散方程:
        #    T_new = T_old + αΔt[(T[i+1]-2T[i]+T[i-1])/Δx²
        #                   + (T[j+1]-2T[j]+T[j-1])/Δy²]
        # 4. 稳定性条件(Fourier数):
        #    αΔt/Δx² ≤ 0.5 且 αΔt/Δy² ≤ 0.5
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

        # 边界节点
        # 顶部边界
        j = ny - 1
        if boundary_type == 3:
            # 第三类边界条件(对流+辐射) - 非稳态形式
            # 公式推导过程:
            # 1. 能量平衡: ρcp(∂T/∂t) = k(∂²T/∂x²) + k(∂²T/∂y²) - h(T-T_inf) - q_rad
            # 2. 离散化处理:
            #    a. 时间导数: (T_new[i,j] - T_old[i,j])/Δt
            #    b. x方向二阶导数: (T_old[i,j+1] - 2T_old[i,j] + T_old[i,j-1])/dx²
            #    c. y方向导数(边界单向差分): (T_old[i,j-1] - T_old[i,j])/dy²
            #    d. 对流项: h(T_old[i,j] - T_inf)
            #    e. 辐射项: q_rad = σε(T_old[i,j]^4 - T_inf^4)
            # 3. 离散平衡方程:
            #    ρcp(T_new[i,j] - T_old[i,j])/Δt*dx*dy =
            #      k[(T_old[i,j+1] - 2T_old[i,j] + T_old[i,j-1])/dx*dy
            #      + k(T_old[i,j-1] - T_old[i,j])/dy*dx
            #      - [h(T_old[i,j] - T_inf) + q_rad]*dx
            # 4. 显式离散方程:
            #    T_new[i,j] = T_old[i,j] + αΔt[(T_old[i,j+1] - 2T_old[i,j] + T_old[i,j-1])/dx²
            #                  + (T_old[i,j-1] - T_old[i,j])/dy²
            #                  - (h(T_old[i,j] - T_inf) + q_rad)/(k *dy)]
            # 2025年5月22日18:31:49
            from boundary_condition import HeatTransferCalculator

            calculator = HeatTransferCalculator()
            for i in range(1, nx - 1):
                k = get_conductivity(T[j, i])
                # 计算辐射热流密度(W/m²)
                q_rad = (
                    calculator.air_cooling_heat_flux(
                        T_s=T_old[j, i], T_a=T_inf_top, emissivity=sigma
                    )
                    * 1000
                )
                # 总热流 = 对流 + 辐射 (W/m²)
                q_total = h_top * (T_old[j, i] - T_inf_top) + q_rad
                # 边界节点温度计算公式(考虑三个方向导热)
                rho = get_density(T_old[i, j])
                cp = get_specific_heat(T_old[i, j])
                alpha = k / (rho * cp)
                T[i, j] = T_old[i, j] + alpha * dt * (
                    (T_old[i, j + 1] - 2 * T_old[i, j] + T_old[i, j - 1]) / dx**2
                    + (T_old[i, j - 1] - T_old[i, j]) / dy**2
                    - (h_top * (T_old[i, j] - T_inf_top) + q_rad) / (k * dy)
                )
                ## 20250522 下班，处理到这里
        else:
            # 第二类边界条件(给定热流密度) - 非稳态形式
            # 公式推导过程:
            # 1. 能量平衡: ρcp(∂T/∂t) = k(∂²T/∂x²) + k(∂T/∂y) - q_top
            # 2. 离散化处理:
            #    a. 时间导数: (T_new[i,j] - T_old[i,j])/Δt
            #    b. x方向二阶导数: (T_old[i,j+1] - 2T_old[i,j] + T_old[i,j-1])/dx²
            #    c. y方向一阶导数: (T_old[i,j] - T_old[i,j-1])/dy²
            # 3. 显式离散方程:
            #    T_new[i,j] = T_old[i,j] + αΔt[(T_old[i,j+1] - 2T_old[i,j] + T_old[i,j-1])/dx²
            #                  + (T_old[i,j] - T_old[i,j-1])/dy² - q_top/(ρcp dy)]
            # 3. 离散方程:
            #    T[j,i] = T_old[j,i] + (αΔt)[(T_old[j,i+1] - 2T_old[j,i] + T_old[j,i-1])/dx²
            #              + (T_old[j,i] - T_old[j-1,i])/dy² + q_top/(k dy)]

            # 更新边界节点温度(非稳态形式)
            for i in range(1, nx - 1):
                k = get_conductivity(T_old[j, i])  # 使用上一时间步的温度计算导热系数
                rho = get_density(T_old[j, i])
                cp = get_specific_heat(T_old[j, i])
                alpha = k / (rho * cp)  # 热扩散系数

                # 非稳态边界条件离散方程
                T[i, j] = T_old[i, j] + alpha * dt * (
                    (T_old[i, j + 1] - 2 * T_old[i, j] + T_old[i, j - 1])
                    / dx**2  # x方向
                    + (T_old[i, j] - T_old[i, j - 1]) / dy**2  # y方向
                    - (calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B) * 1000)
                    / (rho * cp * dy)  # 热流密度项
                    + (calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B) * 1000)
                    / (k * dy)  # 计算当前时间的热流密度并转换为W/m²
                )

        # 右侧边界
        i = nx - 1
        if boundary_type == 3:
            # 第三类边界条件 - 同时考虑对流和辐射
            # 公式推导过程:
            # 1. 能量平衡: 导热进入节点的热量 = 对流+辐射带走的热量
            #    -k*(T[i,j]-T[i-1,j])/dx = h*(T[i,j]-T_inf) + q_rad
            # 2. 离散化处理:
            #    a. x方向导数: (T_old[i-1,j] - T_old[i,j])/dx
            #    b. 对流项: h*(T_old[i,j] - T_inf)
            #    c. 辐射项: q_rad = σε(T_old[i,j]^4 - T_inf^4)
            # 3. 整理得到:
            #    k*T_old[i-1,j]/dx = (k/dx + h)*T_new[i,j] - h*T_inf - q_rad
            # 4. 解出T_new[i,j]:
            #    T_new[i,j] = (k*T_old[i-1,j] + (h*T_inf + q_rad)*dx) / (k + h*dx)
            for j in range(1, ny - 1):
                k = get_conductivity(T[i, j])
                # 计算辐射热流密度(W/m²)
                q_rad = (
                    calculator.air_cooling_heat_flux(
                        T_s=T_old[i, j], T_a=T_inf_right, emissivity=sigma
                    )
                    * 1000
                )
                # 总热流 = 对流 + 辐射 (W/m²)
                q_total = h_right * (T_old[i, j] - T_inf_right) + q_rad
                # 边界节点温度计算公式
                T[i, j] = (k * T[i - 1, j] + q_total * dx) / (k + h_right * dx)
        else:
            # 第二类边界条件 - 非稳态形式
            # 公式推导过程:
            # 1. 能量平衡: ρcp(∂T/∂t) = k(∂²T/∂x²) + k(∂²T/∂y²) + q_right
            # 2. 离散化处理:
            #    a. 时间导数: (T_new - T_old)/Δt
            #    b. x方向一阶导数: (T[j,i] - T[j,i-1])/dx
            #    c. y方向二阶导数: (T[j+1,i] - 2T[j,i] + T[j-1,i])/dy²
            # 3. 离散方程:
            #    T[j,i] = T_old[j,i] + (αΔt)[(T_old[j,i] - T_old[j,i-1])/dx²
            #              + (T_old[j+1,i] - 2T_old[j,i] + T_old[j-1,i])/dy²
            #              + q_right/(k dx)]  # 直接使用输入的热流密度

            for j in range(1, ny - 1):
                k = get_conductivity(T_old[j, i])  # 使用上一时间步的温度计算导热系数
                rho = get_density(T_old[j, i])
                cp = get_specific_heat(T_old[j, i])
                alpha = k / (rho * cp)  # 热扩散系数

                # 非稳态边界条件离散方程
                T[j, i] = T_old[j, i] + alpha * dt * (
                    (T_old[j, i] - T_old[j, i - 1]) / dx**2  # x方向
                    + (T_old[j + 1, i] - 2 * T_old[j, i] + T_old[j - 1, i])
                    / dy**2  # y方向
                    + (calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B) * 1000)
                    / (k * dx)  # 计算当前时间的热流密度并转换为W/m²
                )

        # 角点处理
        k = get_conductivity(T[ny - 1, nx - 1])
        if boundary_type == 3:
            # 角点处理 - 同时考虑对流和辐射
            # 公式推导过程:
            # 1. 能量平衡: 从两个方向进入节点的导热 = 两个方向的对流+辐射
            #    k*(T[j,i-1]-T[j,i])/dx + k*(T[j-1,i]-T[j,i])/dy =
            #      (h_top*(T[j,i]-T_inf_top) + q_rad_top)*dy +
            #      (h_right*(T[j,i]-T_inf_right) + q_rad_right)*dx
            # 2. 整理得到:
            #    k*(T[j,i-1]/dx + T[j-1,i]/dy) =
            #      T[j,i]*(k/dx + k/dy + h_top*dy + h_right*dx) -
            #      (h_top*T_inf_top*dy + h_right*T_inf_right*dx +
            #       q_rad_top*dy + q_rad_right*dx)
            # 3. 解出T[j,i]:
            #    T[j,i] = [k*(T[j,i-1] + T[j-1,i]) +
            #             (q_total_top*dy + q_total_right*dx)] /
            #            (2k + h_top*dy + h_right*dx)

            # 计算顶部辐射热流(W/m²)
            q_rad_top = (
                calculator.air_cooling_heat_flux(
                    T_s=T_old[ny - 1, nx - 1], T_a=T_inf_top, emissivity=sigma
                )
                * 1000
            )
            # 计算右侧辐射热流(W/m²)
            q_rad_right = (
                calculator.air_cooling_heat_flux(
                    T_s=T_old[ny - 1, nx - 1], T_a=T_inf_right, emissivity=sigma
                )
                * 1000
            )
            # 总热流 = 对流 + 辐射 (W/m²)
            q_total_top = h_top * (T_old[ny - 1, nx - 1] - T_inf_top) + q_rad_top
            q_total_right = (
                h_right * (T_old[ny - 1, nx - 1] - T_inf_right) + q_rad_right
            )

            # 角点温度计算公式
            T[ny - 1, nx - 1] = (
                k * (T[ny - 1, nx - 2] + T[ny - 2, nx - 1])
                + q_total_top * dy
                + q_total_right * dx
            ) / (2 * k + h_top * dy + h_right * dx)
        else:
            # 非稳态角点条件离散方程
            # 公式推导过程:
            # 1. 能量平衡: ρcp(∂T/∂t) = k(∂²T/∂x²) + k(∂²T/∂y²) + q_top + q_right
            # 2. 离散化处理:
            #    a. 时间导数: (T_new - T_old)/Δt
            #    b. x方向一阶导数: (T[j,i] - T[j,i-1])/dx
            #    c. y方向一阶导数: (T[j,i] - T[j-1,i])/dy
            # 3. 离散方程:
            #    T[j,i] = T_old[j,i] + (αΔt)[(T_old[j,i] - T_old[j,i-1])/dx²
            #              + (T_old[j,i] - T_old[j-1,i])/dy²
            #              + q_top/(k dy) + q_right/(k dx)]  # 直接使用输入的热流密度
            k = get_conductivity(
                T_old[ny - 1, nx - 1]
            )  # 使用上一时间步的温度计算导热系数
            rho = get_density(T_old[ny - 1, nx - 1])
            cp = get_specific_heat(T_old[ny - 1, nx - 1])
            alpha = k / (rho * cp)  # 热扩散系数

            T[ny - 1, nx - 1] = T_old[ny - 1, nx - 1] + alpha * dt * (
                (T_old[ny - 1, nx - 1] - T_old[ny - 1, nx - 2]) / dx**2  # x方向
                + (T_old[ny - 1, nx - 1] - T_old[ny - 2, nx - 1]) / dy**2  # y方向
                + (calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B) * 1000)
                / (k * dy)  # 计算当前时间的热流密度并转换为W/m²
                + (calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B) * 1000)
                / (k * dx)  # 计算当前时间的热流密度并转换为W/m²
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
    """使用相态相关物性计算的二维瞬态热传导求解器"""
    from boundary_condition import HeatTransferCalculator

    calculator = HeatTransferCalculator()

    # 从初始温度场获取网格尺寸
    if not isinstance(initial_temp, np.ndarray):
        initial_temp = np.array(initial_temp)
    nx, ny = initial_temp.shape
    # 网格生成
    dx = Lx / nx
    dy = Ly / ny
    print(f"Lx: {Lx} Ly: {Ly}")
    print(f"nx: {nx} ny: {ny}")
    print(f"dx: {dx} dy: {dy}")
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(y, x)
    print(f"mesh.shape: {X.shape}")

    # 初始热扩散率检查（使用初始温度场的平均温度）
    avg_temp = np.mean(initial_temp)
    k_init = lamda_cal(avg_temp, 0, Ts, Tl, 0, props)  # 位置参数设为0
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
    print("收敛性检测通过……")
    # 时间步进
    n_steps = int(total_time / dt)
    time_history = []
    temp_history = []

    for n in range(n_steps):
        T_old = T.copy()

        # 边界条件处理
        # 底部 (对称边界)
        T[:, 0] = T[:, 1]  # 底部边界(原左侧)
        # 左侧 (对称边界)
        T[0, :] = T[1, :]  # 左侧边界(原底部)
        print("对称边界未出错……")
        # 内部节点 - 显式离散
        for i in range(1, nx - 2):
            # print(f"x内部边界{i}节点未出错……")
            for j in range(1, ny - 2):
                print(f"y内部边界{i},{j}节点未出错……")
                # 获取当前温度下的物性参数(使用相态相关计算)
                k = lamda_cal(
                    T_old[i, j], (i + j) / 2, Ts, Tl, 0, props
                )  # 位置参数设为(i+j)/2
                rho = rho_cal(T_old[i, j], Ts, Tl, props)
                cp = cp_cal(T_old[i, j], Ts, Tl, props)
                alpha = k / (rho * cp)  # 热扩散系数

                # 显式格式温度更新
                T[i, j] = T_old[i, j] + alpha * dt * (
                    (T_old[i + 1, j] - 2 * T_old[i, j] + T_old[i - 1, j]) / dx**2
                    + (T_old[i, j + 1] - 2 * T_old[i, j] + T_old[i, j - 1]) / dy**2
                )
        print("内部节点数组未出界")
        # 边界节点处理(与原始函数相同，只是物性计算方式不同)
        # 顶部边界
        j = ny - 1
        if boundary_type == 3:
            for i in range(1, nx - 1):
                k = lamda_cal(T[j, i], (i + j) / 2, Ts, Tl, 0, props)
                q_rad = (
                    calculator.air_cooling_heat_flux(
                        T_s=T_old[j, i], T_a=T_inf_top, emissivity=sigma
                    )
                    * 1000
                )
                q_total = h_top * (T_old[j, i] - T_inf_top) + q_rad
                T[j, i] = (
                    k * (T[j, i + 1] + T[j, i - 1]) / dx**2
                    + k * T[j - 1, i] / dy**2
                    + q_total / dy
                ) / (2 * k / dx**2 + k / dy**2 + h_top / dy)
        else:
            for i in range(1, nx - 1):
                k = lamda_cal(T_old[j, i], (i + j) / 2, Ts, Tl, 0, props)
                rho = rho_cal(T_old[j, i], Ts, Tl, props)
                cp = cp_cal(T_old[j, i], Ts, Tl, props)
                alpha = k / (rho * cp)
                T[j, i] = T_old[j, i] + alpha * dt * (
                    (T_old[j, i + 1] - 2 * T_old[j, i] + T_old[j, i - 1]) / dx**2
                    + (T_old[j, i] - T_old[j - 1, i]) / dy**2
                    + (calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B) * 1000)
                    / (k * dy)
                )

        # 右侧边界
        i = nx - 1
        if boundary_type == 3:
            for j in range(1, ny - 2):
                k = lamda_cal(T[j, i], (i + j) / 2, Ts, Tl, 0, props)
                q_rad = (
                    calculator.air_cooling_heat_flux(
                        T_s=T_old[j, i], T_a=T_inf_right, emissivity=sigma
                    )
                    * 1000
                )
                q_total = h_right * (T_old[j, i] - T_inf_right) + q_rad
                T[j, i] = (k * T[j, i - 1] + q_total * dx) / (k + h_right * dx)
        else:
            for j in range(1, ny - 1):
                k = lamda_cal(T_old[j, i], (i + j) / 2, Ts, Tl, 0, props)
                rho = rho_cal(T_old[j, i], Ts, Tl, props)
                cp = cp_cal(T_old[j, i], Ts, Tl, props)
                alpha = k / (rho * cp)
                T[j, i] = T_old[j, i] + alpha * dt * (
                    (T_old[j, i] - T_old[j, i - 1]) / dx**2
                    + (T_old[j + 1, i] - 2 * T_old[j, i] + T_old[j - 1, i]) / dy**2
                    + (calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B) * 1000)
                    / (k * dx)
                )

        # 角点处理
        k = lamda_cal(T[ny - 1, nx - 1], (nx - 1 + ny - 1) / 2, Ts, Tl, 0, props)
        if boundary_type == 3:
            q_rad_top = (
                calculator.air_cooling_heat_flux(
                    T_s=T_old[ny - 1, nx - 1], T_a=T_inf_top, emissivity=sigma
                )
                * 1000
            )
            q_rad_right = (
                calculator.air_cooling_heat_flux(
                    T_s=T_old[ny - 1, nx - 1], T_a=T_inf_right, emissivity=sigma
                )
                * 1000
            )
            q_total_top = h_top * (T_old[ny - 1, nx - 1] - T_inf_top) + q_rad_top
            q_total_right = (
                h_right * (T_old[ny - 1, nx - 1] - T_inf_right) + q_rad_right
            )
            T[ny - 1, nx - 1] = (
                k * (T[ny - 1, nx - 2] + T[ny - 2, nx - 1])
                + q_total_top * dy
                + q_total_right * dx
            ) / (2 * k + h_top * dy + h_right * dx)
        else:
            rho = rho_cal(T_old[ny - 1, nx - 1], Ts, Tl, props)
            cp = cp_cal(T_old[ny - 1, nx - 1], Ts, Tl, props)
            alpha = k / (rho * cp)
            T[ny - 1, nx - 1] = T_old[ny - 1, nx - 1] + alpha * dt * (
                (T_old[ny - 1, nx - 1] - T_old[ny - 1, nx - 2]) / dx**2
                + (T_old[ny - 1, nx - 1] - T_old[ny - 2, nx - 1]) / dy**2
                + (calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B) * 1000)
                / (k * dy)
                + (calculator.mold_heat_flux(n * dt, A=q_k_A, B=q_k_B) * 1000)
                / (k * dx)
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
