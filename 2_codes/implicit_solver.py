import numpy as np
from scipy.sparse import diags, linalg
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ImplicitSolver:
    def __init__(
        self,
        size_x=0.15,
        size_y=0.15,
        k=30,
        rho=7800,
        cp=500,
        initial_temp=1550,
        heat_flux_top=2000,
        heat_flux_bottom=2000,
        heat_flux_left=2000,
        heat_flux_right=2000,
        dx=0.01,
        dy=0.01,
    ):
        """
        初始化钢坯参数
        :param size_x: x方向尺寸(m)
        :param size_y: y方向尺寸(m)
        :param k: 导热系数(W/(m·K))
        :param rho: 密度(kg/m^3)
        :param cp: 比热容(J/(kg·K))
        :param initial_temp: 初始温度(°C)
        :param heat_flux_top: 顶部热流密度(W/m^2)
        :param heat_flux_bottom: 底部热流密度(W/m^2)
        :param heat_flux_left: 左侧热流密度(W/m^2)
        :param heat_flux_right: 右侧热流密度(W/m^2)
        :param dx: x方向空间步长(m)
        :param dy: y方向空间步长(m)
        """
        self.size_x = size_x
        self.size_y = size_y
        self.k = k
        self.rho = rho
        self.cp = cp
        self.initial_temp = initial_temp
        self.heat_flux_top = heat_flux_top
        self.heat_flux_bottom = heat_flux_bottom
        self.heat_flux_left = heat_flux_left
        self.heat_flux_right = heat_flux_right

        # 计算网格划分
        self.nx = int(size_x / dx) + 1
        self.ny = int(size_y / dy) + 1
        self.dx = dx
        self.dy = dy

        # 初始化温度场
        self.T = np.ones((self.nx, self.ny)) * initial_temp

    def set_mesh(self, dx, dy):
        """设置空间步长(m)"""
        self.dx = dx
        self.dy = dy
        self.nx = int(self.size_x / dx) + 1
        self.ny = int(self.size_y / dy) + 1
        self.T = np.ones((self.nx, self.ny)) * self.initial_temp

    def solve(self, dt=0.1, total_time=50):
        """
        求解温度分布
        :param dt: 时间步长(s)
        :param total_time: 总时间(s)
        """
        alpha = self.k / (self.rho * self.cp)  # 热扩散系数
        steps = int(total_time / dt)

        # 创建系数矩阵
        n = self.nx * self.ny
        main_diag = np.ones(n) * (
            1 + 2 * alpha * dt / self.dx**2 + 2 * alpha * dt / self.dy**2
        )

        # 初始化右端向量
        b = self.T.flatten()

        # 设置边界条件
        self._apply_boundary_conditions(main_diag, b, alpha, dt)

        # 构建稀疏矩阵
        diagonals = {
            -self.nx: np.ones(n - self.nx) * (-alpha * dt / self.dy**2),  # 下对角线
            -1: np.ones(n - 1) * (-alpha * dt / self.dx**2),  # 左对角线
            0: main_diag,  # 主对角线
            1: np.ones(n - 1) * (-alpha * dt / self.dx**2),  # 右对角线
            self.nx: np.ones(n - self.nx) * (-alpha * dt / self.dy**2),  # 上对角线
        }

        A = diags(list(diagonals.values()), list(diagonals.keys()), format="csr")

        # 时间步进
        steps = int(total_time / dt)
        for _ in range(steps):
            self.T = linalg.spsolve(A, b).reshape((self.nx, self.ny))
            b = self.T.flatten()  # 更新右端向量
            self._apply_boundary_conditions(main_diag, b, alpha, dt)  # 更新边界条件

    def _apply_boundary_conditions(self, main_diag, b, alpha, dt):
        """应用边界条件"""
        for i in range(self.nx):
            for j in range(self.ny):
                idx = i * self.ny + j
                if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1:
                    # 边界节点
                    if (i == 0 or i == self.nx - 1) and (j == 0 or j == self.ny - 1):
                        # 角节点 - 取相邻两边的平均值
                        main_diag[idx] = (
                            1 + alpha * dt / self.dx**2 + alpha * dt / self.dy**2
                        )
                        if i == 0 and j == 0:  # 左下角
                            b[idx] = (
                                (self.heat_flux_left + self.heat_flux_bottom)
                                * dt
                                / (2 * self.k)
                            )
                        elif i == 0 and j == self.ny - 1:  # 左上角
                            b[idx] = (
                                (self.heat_flux_left + self.heat_flux_top)
                                * dt
                                / (2 * self.k)
                            )
                        elif i == self.nx - 1 and j == 0:  # 右下角
                            b[idx] = (
                                (self.heat_flux_right + self.heat_flux_bottom)
                                * dt
                                / (2 * self.k)
                            )
                        else:  # 右上角
                            b[idx] = (
                                (self.heat_flux_right + self.heat_flux_top)
                                * dt
                                / (2 * self.k)
                            )
                    else:
                        # 边节点
                        main_diag[idx] = (
                            1
                            + 2 * alpha * dt / self.dx**2
                            + 2 * alpha * dt / self.dy**2
                        )
                        if i == 0:  # 左边界
                            b[idx] = self.heat_flux_left * dt / self.k
                        elif i == self.nx - 1:  # 右边界
                            b[idx] = self.heat_flux_right * dt / self.k
                        elif j == 0:  # 下边界
                            b[idx] = self.heat_flux_bottom * dt / self.k
                        else:  # 上边界
                            b[idx] = self.heat_flux_top * dt / self.k

    def plot_temperature(self):
        """使用plotly绘制交互式温度分布图"""
        try:
            if not hasattr(self, "T") or self.T.size == 0:
                raise ValueError("温度场数据未初始化，请先运行solve()方法")

            # 创建物理坐标网格
            x_coords = np.linspace(0, (self.T.shape[1] - 1) * self.dx, self.T.shape[1])
            y_coords = np.linspace(0, (self.T.shape[0] - 1) * self.dy, self.T.shape[0])

            fig = go.Figure(
                data=go.Heatmap(
                    z=self.T,
                    x=x_coords,
                    y=y_coords,
                    colorscale="jet",
                    hoverongaps=False,
                    hovertemplate=(
                        "X: %{x:.3f}m<br>"
                        "Y: %{y:.3f}m<br>"
                        "温度: %{z:.1f}°C<br>"
                        "<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                title=f"温度分布 (最高: {self.T.max():.1f}°C, 最低: {self.T.min():.1f}°C)",
                xaxis_title="X坐标 (m)",
                yaxis_title="Y坐标 (m)",
                width=1000,
                height=800,
                autosize=False,
                margin=dict(l=100, r=100, b=100, t=100),
                xaxis=dict(scaleanchor="y", constrain="domain"),
                yaxis=dict(constrain="domain"),
            )

            fig.update_coloraxes(colorbar=dict(title="温度 (°C)"))

            fig.show()

        except Exception as e:
            print(f"绘图错误: {str(e)}")
            raise


def convergence_test():
    """测试不同时间步长dt对结果的影响"""
    solver = ImplicitSolver()
    solver.set_mesh(0.005, 0.005)

    dts = np.linspace(1, 0.01, 30)  # 从1到0.01取20个点
    corner_temps = []

    for dt in dts:
        solver.solve(dt=dt, total_time=10)
        corner_temps.append(solver.T[0, 0])  # 记录左下角温度

    # 绘制结果
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dts, y=corner_temps, mode="lines+markers", name="角点温度")
    )

    fig.update_layout(
        title="时间步长dt对角点温度的影响(总时间=10s)",
        xaxis_title="时间步长dt(s)",
        yaxis_title="角点温度(°C)",
        xaxis_type="log",
        xaxis=dict(autorange="reversed"),
        width=800,
        height=500,
    )
    fig.show()


if __name__ == "__main__":
    solver = ImplicitSolver()
    solver.set_mesh(0.001, 0.001)
    solver.solve(dt=0.05, total_time=10)
    solver.plot_temperature()
    # convergence_test()
