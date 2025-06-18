import numpy as np
from scipy.sparse import diags, lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class HeatSolver2D:
    """二维热传导隐式求解器(时间向后差分)"""

    def __init__(self, Lx=1.0, Ly=1.0, Nx=51, Ny=51, alpha=1e-5, k=1.0):
        """初始化求解器参数"""
        self.Lx, self.Ly = Lx, Ly  # 区域尺寸(m)
        self.Nx, self.Ny = Nx, Ny  # 网格数
        self.dx, self.dy = Lx / (Nx - 1), Ly / (Ny - 1)  # 网格步长(m)
        self.alpha = alpha  # 热扩散率(m²/s)
        self.k = k  # 导热系数(W/m·K)

        # 初始化温度场(默认为0)
        self.T = np.zeros((Nx, Ny))

    def set_boundary_conditions(self, q_left=0, q_right=0, q_bottom=0, q_top=0):
        """设置第二类边界条件(热流密度)"""
        self.q_left = q_left  # 左边界(W/m²)
        self.q_right = q_right  # 右边界(W/m²)
        self.q_bottom = q_bottom  # 下边界(W/m²)
        self.q_top = q_top  # 上边界(W/m²)

    def set_initial_temperature(self, initial_temp=None, uniform_value=0.0):
        """设置初始温度场

        参数:
            initial_temp: 自定义温度场数组(需与网格尺寸匹配)
            uniform_value: 统一初始温度值(当initial_temp为None时使用)
        """
        if initial_temp is not None:
            if not hasattr(initial_temp, "shape") or initial_temp.shape != (
                self.Nx,
                self.Ny,
            ):
                raise ValueError(
                    f"初始温度场需为Numpy数组且尺寸为({self.Nx}, {self.Ny})"
                )
            self.T = initial_temp.copy()
        else:
            self.T.fill(uniform_value)

    def _build_matrix(self, dt):
        """构建系数矩阵A和右端向量b"""
        N = self.Nx * self.Ny
        A = lil_matrix((N, N))
        b = np.zeros(N)

        # 主对角线系数
        diag_coeff = 1 + 2 * self.alpha * dt * (1 / self.dx**2 + 1 / self.dy**2)
        diag_indices = np.arange(N)
        A[diag_indices, diag_indices] = diag_coeff

        # 相邻节点系数
        x_adj = self.alpha * dt / self.dx**2
        y_adj = self.alpha * dt / self.dy**2

        # 内部节点处理
        for i in range(1, self.Nx - 1):
            for j in range(1, self.Ny - 1):
                idx = i * self.Ny + j
                A[idx, idx - self.Ny] = -x_adj  # 左邻居
                A[idx, idx + self.Ny] = -x_adj  # 右邻居
                A[idx, idx - 1] = -y_adj  # 下邻居
                A[idx, idx + 1] = -y_adj  # 上邻居
                b[idx] = self.T[i, j]  # 前一时刻温度

        # 边界条件处理
        self._apply_boundary_conditions(A, b)

        return csc_matrix(A), b

    def _apply_boundary_conditions(self, A, b):
        """应用边界条件到矩阵A和向量b"""

        # 边界处理函数
        def apply_bc(i, j, boundary_type):
            idx = i * self.Ny + j
            if boundary_type == "left":
                A[idx, idx] = 3 / (2 * self.dx)
                A[idx, idx + self.Ny] = -4 / (2 * self.dx)
                A[idx, idx + 2 * self.Ny] = 1 / (2 * self.dx)
                b[idx] = -self.q_left / self.k
            elif boundary_type == "right":
                A[idx, idx] = 3 / (2 * self.dx)
                A[idx, idx - self.Ny] = -4 / (2 * self.dx)
                A[idx, idx - 2 * self.Ny] = 1 / (2 * self.dx)
                b[idx] = self.q_right / self.k
            elif boundary_type == "bottom":
                A[idx, idx] = 3 / (2 * self.dy)
                A[idx, idx + 1] = -4 / (2 * self.dy)
                A[idx, idx + 2] = 1 / (2 * self.dy)
                b[idx] = -self.q_bottom / self.k
            elif boundary_type == "top":
                A[idx, idx] = 3 / (2 * self.dy)
                A[idx, idx - 1] = -4 / (2 * self.dy)
                A[idx, idx - 2] = 1 / (2 * self.dy)
                b[idx] = self.q_top / self.k

        # 应用边界条件
        for j in range(1, self.Ny - 1):
            apply_bc(0, j, "left")  # 左边界
            apply_bc(self.Nx - 1, j, "right")  # 右边界

        for i in range(1, self.Nx - 1):
            apply_bc(i, 0, "bottom")  # 下边界
            apply_bc(i, self.Ny - 1, "top")  # 上边界

        # 角点处理(简化)
        self._apply_corner_conditions(A, b)

    def _apply_corner_conditions(self, A, b):
        """处理四个角点"""

        def apply_corner(i, j, qx, qy):
            idx = i * self.Ny + j
            A[idx, idx] = 1 / (2 * self.dx) + 1 / (2 * self.dy)

            # 处理x方向邻居 (确保不越界)
            if qx > 0 and i < self.Nx - 1:  # 左边界且有右邻居
                A[idx, idx + self.Ny] = -1 / (2 * self.dx)
            elif qx <= 0 and i > 0:  # 右边界且有左邻居
                A[idx, idx - self.Ny] = 1 / (2 * self.dx)

            # 处理y方向邻居 (确保不越界)
            if qy > 0 and j < self.Ny - 1:  # 下边界且有上邻居
                A[idx, idx + 1] = -1 / (2 * self.dy)
            elif qy <= 0 and j > 0:  # 上边界且有下邻居
                A[idx, idx - 1] = 1 / (2 * self.dy)

            b[idx] = (qx + qy) / (2 * self.k)

        # 四个角点
        apply_corner(0, 0, self.q_left, self.q_bottom)  # 左下
        apply_corner(self.Nx - 1, 0, -self.q_right, self.q_bottom)  # 右下
        apply_corner(0, self.Ny - 1, self.q_left, -self.q_top)  # 左上
        apply_corner(self.Nx - 1, self.Ny - 1, -self.q_right, -self.q_top)  # 右上

    def solve(self, dt=1.0, total_time=1000.0, save_interval=10):
        """执行求解过程

        参数:
            dt: 时间步长(s)
            total_time: 总模拟时间(s)
            save_interval: 保存间隔(时间步数)

        返回:
            time_steps: 保存的时间步列表
            temperature_history: 各时间步的温度场
        """
        num_steps = int(total_time / dt)
        temperature_history = []
        time_steps = []

        for step in range(num_steps):
            A, b = self._build_matrix(dt)
            T_flat = spsolve(A, b)
            self.T = T_flat.reshape((self.Nx, self.Ny))

            # 定期保存结果
            if step % save_interval == 0:
                temperature_history.append(self.T.copy())
                time_steps.append(step * dt)
                print(f"Step {step}/{num_steps}, Max Temp: {np.max(self.T):.2f}°C")

        return time_steps, temperature_history

    @staticmethod
    def visualize(time_steps, temperature_history, output_file="heatmap.html"):
        """使用Plotly可视化温度场随时间变化

        参数:
            time_steps: 时间步列表
            temperature_history: 温度场历史数据
            output_file: 输出HTML文件名
        """
        fig = make_subplots(rows=1, cols=1)

        # 创建动画帧
        frames = []
        for i, (t, T) in enumerate(zip(time_steps, temperature_history)):
            frames.append(
                go.Frame(
                    data=[go.Heatmap(z=T, colorscale="inferno")],
                    name=f"t={t:.1f}s",
                    traces=[0],
                )
            )

        # 初始帧
        fig.add_trace(
            go.Heatmap(
                z=temperature_history[0],
                colorscale="inferno",
                colorbar=dict(title="Temperature (°C)"),
            )
        )

        # 更新布局
        fig.update_layout(
            title="2D Heat Transfer Simulation",
            xaxis_title="x",
            yaxis_title="y",
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {"frame": {"duration": 100, "redraw": True}},
                            ],
                        }
                    ],
                }
            ],
        )

        fig.frames = frames
        fig.write_html(output_file)
        print(f"Visualization saved to {output_file}")


def test_heat_solver(
    Lx=0.15,
    Ly=0.15,
    Nx=51,
    Ny=51,
    alpha=6e-6,
    k=30.0,
    q_left=2000,
    q_right=2000,
    q_bottom=2000,
    q_top=2000,
    dt=0.01,
    total_time=60,
    save_interval=10,
    output_file="heatmap.html",
    initial_temp=np.ones((51, 51)) * 1500.0,  # 初始温度场
    uniform_temp=0.0,
):
    """测试热传导求解器

    参数:
        Lx, Ly: 区域尺寸(m)
        Nx, Ny: 网格数
        alpha: 热扩散率(m²/s)
        k: 导热系数(W/m·K)
        q_*: 各边界热流密度(W/m²)
        dt: 时间步长(s)
        total_time: 总模拟时间(s)
        save_interval: 保存间隔(时间步数)
        output_file: 输出HTML文件名
        initial_temp: 自定义初始温度场数组
        uniform_temp: 统一初始温度值(当initial_temp为None时使用)
    """
    # 创建求解器实例
    solver = HeatSolver2D(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, alpha=alpha, k=k)

    # 设置初始温度
    solver.set_initial_temperature(
        initial_temp=initial_temp, uniform_value=uniform_temp
    )

    # 设置边界条件
    solver.set_boundary_conditions(
        q_left=q_left, q_right=q_right, q_bottom=q_bottom, q_top=q_top
    )

    # 执行求解
    time_steps, temp_history = solver.solve(
        dt=dt, total_time=total_time, save_interval=save_interval
    )

    # 可视化结果
    HeatSolver2D.visualize(time_steps, temp_history, output_file)
    return temp_history


if __name__ == "__main__":
    # 默认测试案例
    # print("运行默认测试案例...")
    # default_result = test_heat_solver()

    # 用户可修改以下参数自定义测试
    print("\n要自定义参数，请修改以下代码:")
    print(
        """
    custom_result = test_heat_solver(
        Lx=2.0, Ly=1.5, Nx=61, Ny=41, alpha=2e-5, k=0.8,
        q_left=300.0, q_right=-100.0, q_bottom=50.0, q_top=-200.0,
        dt=0.5, total_time=500.0, save_interval=5,
        output_file="custom_heatmap.html",
        uniform_temp=1550.0  # 设置初始温度为1550°C
    )
    """
    )
    test_heat_solver()
