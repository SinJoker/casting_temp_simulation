# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math


class SteelTemperatureSimulator:
    """
    钢坯温度场模拟核心计算类
    模拟165x165mm钢坯1/4截面的二维热传导过程
    """

    def __init__(self, space_step=1.0, time_step=0.1):
        """
        初始化模拟参数
        :param space_step: 空间步长(mm)
        :param time_step: 时间步长(s)
        """
        # 材料参数默认值
        self._conductivity = 30.0  # 默认导热系数(W/m·K)
        self._density = 7800.0  # 默认密度(kg/m³)
        self._specific_heat = 500.0  # 默认比热容(J/kg·K)

        # 模拟参数
        self.space_step = space_step / 1000  # 转为米
        self.time_step = time_step
        self.heat_flux = 10000000.0  # 热流密度(W/m²)

        # 计算网格
        self.nx = int(82.5 / space_step) + 1  # 165/2=82.5mm
        self.ny = int(82.5 / space_step) + 1

        # 初始化温度场 (1550℃)
        self.T = np.ones((self.ny, self.nx)) * 1550
        self.H = (
            self.T * self.get_density(self.T) * self.get_specific_heat(self.T)
        )  # 焓场

    def run_simulation(self, total_time=100.0):
        """
        运行温度场模拟
        :param total_time: 总模拟时间(s)
        :return: 温度场历史记录
        """
        # 计算平均热扩散系数
        k = np.mean(self.get_conductivity(self.T))
        rho = np.mean(self.get_density(self.T))
        cp = np.mean(self.get_specific_heat(self.T))
        alpha = k / (rho * cp)
        steps = int(total_time / self.time_step)
        history = []

        for step in range(steps):
            new_H = self.H.copy()

            # 内部节点计算
            for i in range(1, self.ny - 1):
                for j in range(1, self.nx - 1):
                    # 二维热传导方程离散格式
                    dTdx2 = (self.T[i + 1, j] - 2 * self.T[i, j] + self.T[i - 1, j]) / (
                        self.space_step**2
                    )
                    dTdy2 = (self.T[i, j + 1] - 2 * self.T[i, j] + self.T[i, j - 1]) / (
                        self.space_step**2
                    )
                    new_H[i, j] = self.H[i, j] + alpha[()] * self.time_step * (
                        dTdx2 + dTdy2
                    )

                    # 检查稳定性条件 (dt <= dx^2/(4*alpha))
                    max_alpha = np.max(self.get_conductivity(self.T)) / (
                        np.min(self.get_density(self.T))
                        * np.min(self.get_specific_heat(self.T))
                    )
                    stable_dt = (self.space_step**2) / (4 * max_alpha)
                    if self.time_step > stable_dt:
                        print(
                            f"警告: 时间步长{self.time_step}可能过大，建议小于{stable_dt:.4f}以保证稳定性"
                        )

            # 边界条件处理
            self._apply_boundary_conditions(new_H, alpha)

            # 更新温度场
            self.H = new_H
            self.T = self.H / (
                self.get_density(self.T) * self.get_specific_heat(self.T)
            )
            # 限制温度在0℃到初始温度1550℃之间
            self.T = np.clip(self.T, 0, 1550)
            history.append(self.T.copy())

            # 打印进度
            if step % 100 == 0:
                avg_k = np.mean(self.get_conductivity(self.T))
                print(
                    f"进度: {step+1}/{steps} 步, 最高温度: {np.max(self.T):.1f}℃, 平均导热系数: {avg_k:.2f} W/m·K"
                )

        return history

    def _apply_boundary_conditions(self, new_H, alpha):
        """应用各种边界条件"""
        # 左边界 (x=0, 绝热)
        i = 0
        for j in range(1, self.nx - 1):
            # 左边界绝热条件 (dT/dx=0)
            dTdx = 0  # 绝热边界
            dTdy2 = (self.T[i, j + 1] - 2 * self.T[i, j] + self.T[i, j - 1]) / (
                self.space_step**2
            )
            new_H[i, j] = self.H[i, j] + alpha[()] * self.time_step * dTdy2

        # 下边界 (y=0, 绝热)
        j = 0
        for i in range(1, self.ny - 1):
            # 下边界绝热条件 (dT/dy=0)
            dTdx2 = (self.T[i + 1, j] - 2 * self.T[i, j] + self.T[i - 1, j]) / (
                self.space_step**2
            )
            dTdy = 0  # 绝热边界
            new_H[i, j] = self.H[i, j] + alpha[()] * self.time_step * dTdx2

        # 右边界 (x=82.5mm, 对流)
        i = self.ny - 1
        for j in range(1, self.nx - 1):
            new_H[i, j] = (
                self.H[i, j]
                - 2
                * self.heat_flux
                * self.time_step
                / (self.get_density(self.T[i, j]) * self.space_step)
                + alpha[()]
                * self.time_step
                * (
                    (self.T[i, j + 1] - 2 * self.T[i, j] + self.T[i, j - 1])
                    / (self.space_step**2)
                )
            )

        # 上边界 (y=82.5mm, 对流)
        j = self.nx - 1
        for i in range(1, self.ny - 1):
            new_H[i, j] = (
                self.H[i, j]
                - 2
                * self.heat_flux
                * self.time_step
                / (self.get_density(self.T[i, j]) * self.space_step)
                + alpha[()]
                * self.time_step
                * (
                    (self.T[i + 1, j] - 2 * self.T[i, j] + self.T[i - 1, j])
                    / (self.space_step**2)
                )
            )

        # 角点 (x=82.5mm,y=82.5mm)
        i = self.ny - 1
        j = self.nx - 1
        new_H[i, j] = self.H[i, j] - 4 * self.heat_flux * self.time_step / (
            self.get_density(self.T[i, j]) * self.space_step
        )

    def get_conductivity(self, T):
        """获取导热系数(W/m·K)与温度的关系"""
        if isinstance(T, np.ndarray):
            return np.vectorize(self.get_conductivity)(T)

        self._conductivity = (0.45041 - 1.7057 / 10000 * T) * 100

        return self._conductivity  # 默认返回固定值

    def get_density(self, T):
        """获取密度(kg/m³)与温度的关系"""
        if isinstance(T, np.ndarray):
            return np.vectorize(self.get_density)(T)
        self._density = (7.8137 - 3.3481 / 10000 * T) * 1000
        return self._density  # 默认返回固定值

    def get_specific_heat(self, T):
        """获取比热容(J/kg·K)与温度的关系"""
        if isinstance(T, np.ndarray):
            return np.vectorize(self.get_specific_heat)(T)
        self._specific_heat = (0.10971 + 5.4016 / 100000 * T) * 4184

        return self._specific_heat  # 默认返回固定值

    def visualize(self, T=None):
        """可视化温度场"""
        if T is None:
            T = self.T

        # 设置中文字体
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
        plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

        plt.figure(figsize=(10, 8))
        plt.imshow(T, cmap="hot", origin="lower")
        plt.colorbar(label="温度(℃)")
        plt.title("连铸坯1/4截面温度场分布")
        plt.xlabel("宽度方向(mm)")
        plt.ylabel("厚度方向(mm)")
        plt.show()


# 使用示例
if __name__ == "__main__":
    simulator = SteelTemperatureSimulator(space_step=10, time_step=0.3)
    history = simulator.run_simulation(total_time=6000.0)
    simulator.visualize()
