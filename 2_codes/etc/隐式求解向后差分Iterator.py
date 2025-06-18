import numpy as np
from scipy.sparse import diags, lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def solve_2d_heat_transfer():
    # 参数设置
    Lx, Ly = 0.15, 0.15  # 区域尺寸
    Nx, Ny = 101, 101  # 网格数 (奇数便于中心处理)
    dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # 网格步长
    alpha = 1e-5  # 热扩散率
    dt = 0.1  # 时间步长
    total_time = 50.0  # 总时间
    k = 30.0  # 导热系数

    # 边界热流密度 (W/m²) 方向统一规定:
    # 1. 所有边界: 热流密度为正表示热量流出计算域
    # 2. 根据傅里叶定律 q = -k*∇T
    # 3. 对于左/右边界:
    #    - q_left为正表示热量向右流出(∂T/∂x为正)
    #    - q_right为正表示热量向左流出(∂T/∂x为负)
    # 4. 对于上/下边界:
    #    - q_bottom为正表示热量向上流出(∂T/∂y为正)
    #    - q_top为正表示热量向下流出(∂T/∂y为负)
    # 5. 角点处两个方向热流直接相加

    # 默认边界热流值 (正值表示热量流出计算域)
    q_left = 2000.0  # 左边界热流 (W/m²)
    q_right = 0.0  # 右边界热流 (W/m²)
    q_bottom = 0.0  # 下边界热流 (W/m²)
    q_top = 0.0  # 上边界热流 (W/m²)

    # 初始化温度场
    T = np.ones((Nx, Ny)) * 1550  # 初始温度

    # 设置系数矩阵
    N = Nx * Ny
    A = lil_matrix((N, N))
    b = np.zeros(N)

    # 主对角线系数
    diag_coeff = 1 + 2 * alpha * dt * (1 / dx**2 + 1 / dy**2)
    diag_indices = np.arange(N)
    A[diag_indices, diag_indices] = diag_coeff

    # 相邻节点系数
    x_adj = alpha * dt / dx**2
    y_adj = alpha * dt / dy**2

    # 内部节点 (x 和 y 方向的相邻节点)
    for i in range(1, Nx - 2):
        for j in range(1, Ny - 2):
            idx = i * Ny + j
            # 左邻居 (i-1,j)
            A[idx, idx - Ny] = -x_adj
            # 右邻居 (i+1,j)
            A[idx, idx + Ny] = -x_adj
            # 下邻居 (i,j-1)
            A[idx, idx - 1] = -y_adj
            # 上邻居 (i,j+1)
            A[idx, idx + 1] = -y_adj
            b[idx] = T[i, j]  # 来自前一时刻的温度

    # 边界处理函数 - 应用热流边界条件
    # Boundary condition handler - Apply heat flux boundary conditions
    def apply_boundary_condition(i, j, boundary_type):
        idx = i * Ny + j  # 计算矩阵索引 | Compute matrix index

        if boundary_type == "left":
            # 左边界使用前向差分近似温度梯度 | Left boundary uses forward difference
            # ∂T/∂x ≈ (-3T0 + 4T1 - T2)/(2dx) (二阶精度)
            # 根据傅里叶定律 q = -k*∂T/∂x → ∂T/∂x = -q/k
            # 对于左边界，法向为-x方向，所以q_left为正表示流出
            A[idx, idx] = 3 / (2 * dx)  # 对角线系数
            A[idx, idx + Ny] = -4 / (2 * dx)  # T[i+1,j]系数
            A[idx, idx + 2 * Ny] = 1 / (2 * dx)  # T[i+2,j]系数
            b[idx] = +q_left / k  # ∂T/∂x = -q_left/k
            return

        if boundary_type == "right":
            # 后向差分: ∂T/∂x ≈ (3T0 - 4T_{-1} + T_{-2})/(2dx)
            A[idx, idx] = 3 / (2 * dx)
            A[idx, idx - Ny] = -4 / (2 * dx)  # T[i-1,j]
            A[idx, idx - 2 * Ny] = 1 / (2 * dx)  # T[i-2,j]
            b[idx] = -q_right / k  # ∂T/∂x = q_right/k
            return

        if boundary_type == "bottom":
            # ∂T/∂y ≈ (-3T0 + 4T_{j+1} - T_{j+2})/(2dy)
            A[idx, idx] = 3 / (2 * dy)
            A[idx, idx + 1] = -4 / (2 * dy)  # T[i,j+1]
            A[idx, idx + 2] = 1 / (2 * dy)  # T[i,j+2]
            b[idx] = +q_bottom / k  # ∂T/∂y = -q_bottom/k
            return

        if boundary_type == "top":
            # ∂T/∂y ≈ (3T0 - 4T_{j-1} + T_{j-2})/(2dy)
            A[idx, idx] = 3 / (2 * dy)
            A[idx, idx - 1] = -4 / (2 * dy)  # T[i,j-1]
            A[idx, idx - 2] = 1 / (2 * dy)  # T[i,j-2]
            b[idx] = -q_top / k  # ∂T/∂y = q_top/k
            return

    # 统一角点处理方案 - 仅考虑主导热流方向
    def apply_corner_condition(i, j, corner_type):
        idx = i * Ny + j
        area_factor = 0.5  # 更小的控制面积系数

        if corner_type == "bottom_left":
            # 左下角: 仅考虑左边界热流
            A[idx, idx] = 3 / (2 * dx)
            A[idx, idx + Ny] = -4 / (2 * dx)
            A[idx, idx + 2 * Ny] = 1 / (2 * dx)
            # y方向设为绝热
            A[idx, idx + 1] = -1 / dy
            A[idx, idx + 2] = 1 / dy
            b[idx] = area_factor * q_left / k

        elif corner_type == "bottom_right":
            # 右下角: 设为绝热
            A[idx, idx] = 1
            A[idx, idx - 1] = -1  # 连接右侧节点
            b[idx] = 0

        elif corner_type == "top_left":
            # 左上角: 仅考虑左边界热流
            A[idx, idx] = 3 / (2 * dx)
            A[idx, idx + Ny] = -4 / (2 * dx)
            A[idx, idx + 2 * Ny] = 1 / (2 * dx)
            # y方向设为绝热
            A[idx, idx - 1] = -1 / dy
            A[idx, idx - 2] = 1 / dy
            b[idx] = area_factor * q_left / k

        elif corner_type == "top_right":
            # 右上角: 设为绝热
            A[idx, idx] = 1
            A[idx, idx - 1] = -1  # 连接右侧节点
            b[idx] = 0

    # 应用边界条件
    for j in range(1, Ny - 2):
        # 左边界 (i=0)
        apply_boundary_condition(0, j, "left")
        # 右边界 (i=Nx-1)
        apply_boundary_condition(Nx - 1, j, "right")

    for i in range(1, Nx - 2):
        # 下边界 (j=0)
        apply_boundary_condition(i, 0, "bottom")
        # 上边界 (j=Ny-1)
        apply_boundary_condition(i, Ny - 1, "top")

    # 应用角点条件
    apply_corner_condition(0, 0, "bottom_left")  # 左下角
    apply_corner_condition(Nx - 1, 0, "bottom_right")  # 右下角
    apply_corner_condition(0, Ny - 1, "top_left")  # 左上角
    apply_corner_condition(Nx - 1, Ny - 1, "top_right")  # 右上角

    # 转换为CSC格式提高求解效率
    A_csc = csc_matrix(A)

    # 时间推进求解
    num_steps = int(total_time / dt)
    center_temps = []  # 记录中心点温度变化
    for step in range(num_steps):
        # 求解线性方程组
        T_flat = spsolve(A_csc, b)

        # 更新温度场和b向量
        T = T_flat.reshape((Nx, Ny))
        b[diag_indices] = T.flatten()  # 更新b中的时间项

        # 记录中心点温度
        center_temp = T[Nx // 2, Ny // 2]
        center_temps.append(center_temp)

        # 每100步输出进度和能量变化
        if step % 100 == 0:
            total_heat = np.sum(T) * dx * dy
            print(
                f"Step {step}/{num_steps}, Max Temp: {np.max(T):.2f}, \nleft_bottom_temp: {T[0, 0]:.2f}, left_top_temp: {T[0, Ny - 1]:.2f},  right_bottom_temp: {T[Nx - 1, 0]:.2f}, right_top_temp: {T[Nx - 1, Ny - 1]:.2f}, "
                f"\nCenter Temp: {center_temp:.2f}, Total Heat: {total_heat:.2e}"
            )

    # # 绘制中心点温度变化曲线
    # plt.figure()
    # plt.plot(np.linspace(0, total_time, num_steps), center_temps)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Center Temperature (°C)")
    # plt.title("Center Point Temperature Evolution")

    # # 可视化结果
    # plt.figure(figsize=(10, 8))
    # plt.imshow(T, cmap="inferno", extent=[0, Lx, 0, Ly], origin="lower")
    # plt.colorbar(label="Temperature (°C)")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title(f"Temperature Distribution at t = {total_time}s")

    # 绘制热流矢量图
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    qx = -np.gradient(T, dx, axis=0)
    qy = -np.gradient(T, dy, axis=1)

    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, qx, qy, scale=1000, color="gray")
    plt.imshow(T, cmap="inferno", extent=[0, Lx, 0, Ly], origin="lower", alpha=0.7)
    plt.colorbar(label="Temperature (°C)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Heat Flux Vectors")

    # 中心点温度变化
    center_temp = T[Nx // 2, Ny // 2]
    print(f"Center temperature: {center_temp:.2f}°C")

    plt.show()
    return T


# 运行求解
temperature_field = solve_2d_heat_transfer()
