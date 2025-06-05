import matplotlib.pyplot as plt
import numpy as np


def plot_contour(
    Coordx, Coordy, Tem, minX, maxX, minY, maxY, figName, levelsYmin=0, levelsYmax=0
):
    """2D输入"""
    if levelsYmin == 0:
        levelsYmin = np.min(Tem)
    if levelsYmax == 0:
        levelsYmax = np.max(Tem)
    fig, ax = plt.subplots(figsize=(12, 8))
    levels = np.linspace(levelsYmin, levelsYmax, 21)
    cset1 = ax.contourf(
        Coordx, Coordy, Tem, levels, cmap=plt.cm.jet
    )  # 设置cmap为jet，最小值为蓝色，最大为红色

    ax.set_title("Time = " + str(figName) + "s", size=20)
    ax.set_xlim(minX, maxX)
    ax.set_ylim(minY, maxY)
    ax.set_xlabel("X(mm)", size=20)
    ax.set_ylabel("Y(mm)", size=20)
    ax.set_xticks(np.around(np.linspace(minX, maxX, 6), decimals=1))
    ax.set_yticks(np.around(np.linspace(minY, maxY, 6), decimals=1))

    cbar = fig.colorbar(cset1)  # 设置colorbar的刻度，标签
    cbar.set_label("Temperature /K", size=20)
    cbar.set_ticks(np.linspace(levelsYmin, levelsYmax, 11))

    # fig.savefig(figName + ".png", bbox_inches='tight', dpi=150, pad_inches=0.1)
    plt.show()


""" 声明变量 """
nx = 51
ny = 51

thermal_cond, rho, cp = 237, 2700, 880  # 材料为金属铝
a = thermal_cond / (rho * cp)  # 热扩散率

dx = 0.2 / (nx - 1)
dy = 0.2 / (ny - 1)

dt = 0.001  # 限制时间步长，防止计算发散
nt = 100000
print_interval = 2.5

h = 1000000
tf = 300

x = np.linspace(0, 0.2, nx)
y = np.linspace(0, 0.2, ny)
T = np.ones((ny, nx))
X, Y = np.meshgrid(x, y)

""" 初始化 """
T[0, :], T[-1, :], T[:, 0], T[:, -1], T[1:-1, 1:-1] = 900, 900, 900, 900, 900
plot_contour(X, Y, T, 0, 0.2, 0, 0.2, 0, 300, 900)
print("中心温度：", T[int(nx / 2), int(ny / 2)], " K", "时间：", 0 * dt, " s")

""" 数值计算 """
eps = 1e-5
iterStep = 1000

for i in range(nt):
    T_last = T.copy()
    for j in range(iterStep):
        Tn = T.copy()
        # 中心节点
        T[1:-1, 1:-1] = (
            T_last[1:-1, 1:-1]
            + a * dt / (dx * dx) * (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, 0:-2])
            + a * dt / (dy * dy) * (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[0:-2, 1:-1])
        )
        # 上边界-边
        T[-1, 1:-1] = T_last[-1, 1:-1] + (
            h * dx * (tf - T[-1, 1:-1])
            - thermal_cond * (T[-1, 1:-1] - T[-1, 2:]) * dy / (2 * dx)
            - thermal_cond * (T[-1, 1:-1] - T[-1, 0:-2]) * dy / (2 * dx)
            - thermal_cond * (T[-1, 1:-1] - T[-2, 1:-1]) * dx / dy
        ) * dt / (rho * cp * 0.5 * dx * dy)
        # 下边界-边
        T[0, 1:-1] = T_last[0, 1:-1] + (
            h * dx * (tf - T[0, 1:-1])
            - thermal_cond * (T[0, 1:-1] - T[0, 2:]) * dy / (2 * dx)
            - thermal_cond * (T[0, 1:-1] - T[0, 0:-2]) * dy / (2 * dx)
            - thermal_cond * (T[0, 1:-1] - T[1, 1:-1]) * dx / dy
        ) * dt / (rho * cp * 0.5 * dx * dy)
        # 左边界-边
        T[1:-1, 0] = T_last[1:-1, 0] + (
            h * dy * (tf - T[1:-1, 0])
            - thermal_cond * (T[1:-1, 0] - T[2:, 0]) * dx / (2 * dy)
            - thermal_cond * (T[1:-1, 0] - T[0:-2, 0]) * dx / (2 * dy)
            - thermal_cond * (T[1:-1, 0] - T[1:-1, 1]) * dy / dx
        ) * dt / (rho * cp * 0.5 * dx * dy)
        # 右边界-边
        T[1:-1, -1] = T_last[1:-1, -1] + (
            h * dy * (tf - T[1:-1, -1])
            - thermal_cond * (T[1:-1, -1] - T[2:, -1]) * dx / (2 * dy)
            - thermal_cond * (T[1:-1, -1] - T[0:-2, -1]) * dx / (2 * dy)
            - thermal_cond * (T[1:-1, -1] - T[1:-1, -2]) * dy / dx
        ) * dt / (rho * cp * 0.5 * dx * dy)
        # 左上边界-点
        T[-1, 0] = T_last[-1, 0] + (
            h * 0.5 * dy * (tf - T[-1, 0])
            + h * 0.5 * dx * (tf - T[-1, 0])
            - thermal_cond * (T[-1, 0] - T[-2, 0]) * dx / (2 * dy)
            - thermal_cond * (T[-1, 0] - T[-1, 1]) * dy / (2 * dx)
        ) * dt / (rho * cp * 0.5 * dx * 0.5 * dy)
        # 左下边界-点
        T[0, 0] = T_last[0, 0] + (
            h * 0.5 * dy * (tf - T[0, 0])
            + h * 0.5 * dx * (tf - T[0, 0])
            - thermal_cond * (T[0, 0] - T[1, 0]) * dx / (2 * dy)
            - thermal_cond * (T[0, 0] - T[0, 1]) * dy / (2 * dx)
        ) * dt / (rho * cp * 0.5 * dx * 0.5 * dy)
        # 右上边界-点
        T[-1, -1] = T_last[-1, -1] + (
            h * 0.5 * dy * (tf - T[-1, -1])
            + h * 0.5 * dx * (tf - T[-1, -1])
            - thermal_cond * (T[-1, -1] - T[-2, -1]) * dx / (2 * dy)
            - thermal_cond * (T[-1, -1] - T[-1, -2]) * dy / (2 * dx)
        ) * dt / (rho * cp * 0.5 * dx * 0.5 * dy)
        # 右下边界-点
        T[0, -1] = T_last[0, -1] + (
            h * 0.5 * dy * (tf - T[0, -1])
            + h * 0.5 * dx * (tf - T[0, -1])
            - thermal_cond * (T[0, -1] - T[1, -1]) * dx / (2 * dy)
            - thermal_cond * (T[0, -1] - T[0, -2]) * dy / (2 * dx)
        ) * dt / (rho * cp * 0.5 * dx * 0.5 * dy)

        error = np.max(np.abs(T - Tn))
        if error < eps:
            # print('No.', i + 1, 'Convergence Y, Time = ', (i + 1) * dt, 'iteration = ', j + 1)
            break
        if j == iterStep - 1:
            print(
                "No.",
                i + 1,
                "Convergence N, Time = ",
                (i + 1) * dt,
                "iteration = ",
                j + 1,
            )

    if (i + 1) * dt % print_interval == 0:
        plot_contour(X, Y, T, 0, 0.2, 0, 0.2, (i + 1) * dt, 300, 900)
        print(
            "中心温度：",
            T[int(nx / 2), int(ny / 2)],
            " K",
            "时间：",
            (i + 1) * dt,
            " s",
        )
    if T[int(nx / 2), int(ny / 2)] <= 450:
        plot_contour(X, Y, T, 0, 0.2, 0, 0.2, (i + 1) * dt, 300, 900)
        print(
            "中心温度：",
            T[int(nx / 2), int(ny / 2)],
            " K",
            "时间：",
            (i + 1) * dt,
            " s",
        )
        break
