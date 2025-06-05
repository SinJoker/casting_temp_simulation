import json
import os
from typing import Any, Dict

import numpy as np


class ModelInitializer:
    def __init__(self):
        self.process_data = None
        self.initialize_data = None
        self.liquid_temp = None
        self.solid_temp = None

    def load_process_data(
        self, file_path: str = "results/process_data.json"
    ) -> Dict[str, Any]:
        """从JSON文件加载工艺参数

        Args:
            file_path: JSON文件路径

        Returns:
            加载的工艺数据

        Raises:
            RuntimeError: 如果文件过大或格式错误
        """
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 10MB
                raise RuntimeError(
                    f"文件过大({file_size/1024/1024:.1f}MB)，请优化数据格式"
                )

            with open(file_path, "r", encoding="utf-8") as f:
                self.process_data = json.load(f)

            # 验证必要字段
            required_fields = [
                "mold_parameters",
                "spray_parameters",
                "air_cooling_parameters",
            ]
            for field in required_fields:
                if field not in self.process_data:
                    raise ValueError(f"缺少必要字段: {field}")

            return self.process_data
        except Exception as e:
            raise RuntimeError(f"加载工艺数据失败: {str(e)}") from e

    def load_const_results(
        self, file_path: str = "results/const_results.json"
    ) -> Dict[str, Any]:
        """从JSON文件加载常量结果

        Args:
            file_path: JSON文件路径

        Returns:
            加载的常量结果数据
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                const_data = json.load(f)
                self.liquid_temp = const_data["liquid_temp"]
                self.solid_temp = const_data["solid_temp"]
            return const_data
        except Exception as e:
            raise RuntimeError(f"加载常量结果失败: {str(e)}") from e

    def calculate_parameters(
        self, grid_size: float = 1.0, time_step: float = 1.0
    ) -> Dict[str, Any]:
        """计算初始化参数

        Args:
            grid_size: 网格尺寸(mm), 默认20mm
            time_step: 时间步长(秒), 默认1秒

        Raises:
            ValueError: 如果网格数超过限制(50x50=2500)
        """
        if not self.process_data:
            raise ValueError("Process data not loaded. Call load_process_data() first.")

        mold = self.process_data["mold_parameters"]
        spray = self.process_data["spray_parameters"]
        air = self.process_data["air_cooling_parameters"]

        # 计算总长度(结晶器+二冷区+空冷区)
        total_length = (
            mold["mold_length"]
            + sum(spray["segment_length"])
            + air["cooling_length"] * 1000
        ) / 1000  # 转换为米

        # 计算网格参数并校验
        nx = int(mold["slab_width"] / grid_size)
        ny = int(mold["slab_thickness"] / grid_size)

        # if nx > 50 or ny > 50:
        #     raise ValueError(
        #         f"网格数{nx}x{ny}过大，请增大网格尺寸。"
        #         f"当前建议: 宽度方向至少{mold['slab_width']/50:.1f}mm, "
        #         f"厚度方向至少{mold['slab_thickness']/50:.1f}mm"
        #     )

        # # 计算换热系数(取二冷区第一个区域的换热系数)
        # h_top = spray["heat_transfer_coefficients"][0]["top"]
        # h_right = spray["heat_transfer_coefficients"][0]["side"]

        # 计算时间步长和拉速
        dt = time_step  # 使用参数传入的时间步长
        casting_speed_mps = mold["casting_speed"] / 60  # 转换为m/s

        # 加载钢种参数(假设使用Q235钢种)
        with open("resources/steel_properties.json", "r", encoding="utf-8") as f:
            steel_props = json.load(f)

        # 计算初始温度场
        casting_temperature = self.process_data["mold_parameters"][
            "casting_temperature"
        ]
        initial_temp_field = casting_temperature

        # 设置不同区域的初始温度
        # 结晶器区域: 全部为液相线温度
        # 二冷区: 根据位置逐渐降低温度
        # 空冷区: 环境温度

        # 构建分段参数
        segments = []

        # 结晶器段(第二类边界条件)
        segments.append(
            {
                "type": "mold",
                "zone_id": 0,
                "boundary_type": 2,
                "q_k_A": mold["heat_flux_a"],
                "q_k_B": mold["heat_flux_b"],
                "h_top": 0.0,
                "h_right": 0.0,
                "T_inf_top": 0.0,
                "T_inf_right": 0.0,
                "sigma": 0.0,
                "initial_temp_field": initial_temp_field,
                "time": (mold["mold_length"]) / casting_speed_mps,
            }
        )

        # 二冷区各段(第三类边界条件)
        for i in range(spray["zone_count"]):
            segment_length_m = spray["segment_length"][i]
            segments.append(
                {
                    "type": "spray",
                    "zone_id": i + 1,
                    "boundary_type": 3,
                    "q_k_A": 0.0,
                    "q_k_B": 0.0,
                    "h_top": spray["heat_transfer_coefficients"][i]["top"],
                    "h_right": spray["heat_transfer_coefficients"][i]["side"],
                    "T_inf_top": spray["water_temps"][i],
                    "T_inf_right": spray["water_temps"][i],
                    "sigma": 0.8,
                    "initial_temp_field": None,
                    "time": segment_length_m / casting_speed_mps,
                }
            )

        # 空冷区(第三类边界条件)
        segments.append(
            {
                "type": "air",
                "zone_id": spray["zone_count"] + 1,
                "boundary_type": 3,
                "q_k_A": 0.0,
                "q_k_B": 0.0,
                "h_top": 10.0,
                "h_right": 10.0,
                "T_inf_top": air["ambient_temp"],
                "T_inf_right": air["ambient_temp"],
                "sigma": air["emissivity"],
                "initial_temp_field": None,  # 由前一段传递
                "time": air["cooling_length"] / casting_speed_mps,
            }
        )

        # 计算总时间(各段时间之和)
        total_time = sum(segment["time"] for segment in segments)

        # 加载常量属性
        with open("results/const_results.json", "r", encoding="utf-8") as f:
            const_results = json.load(f)

        self.initialize_data = {
            "global_parameters": {
                "Lx": mold["slab_width"],  # 米
                "Ly": mold["slab_thickness"],  # 米
                "nx": nx,
                "ny": ny,
                "dt": dt,
                "grid_size": grid_size,
                "time_step": time_step,
                "casting_temperature": self.process_data["mold_parameters"][
                    "casting_temperature"
                ],
                "tol": 1e-5,  # 更新为1e-5
                "liquid_temp": self.liquid_temp,
                "solid_temp": self.solid_temp,
                "props": const_results["const_properties"],
            },
            "segments": segments,
        }

        return self.initialize_data

    def save_initialize_data(self, file_path: str = "results/initialize.json"):
        """保存初始化参数到JSON文件，每次写入前会自动清空原有内容

        Args:
            file_path: 保存文件路径

        Raises:
            ValueError: 如果没有初始化数据
            RuntimeError: 如果保存失败
        """
        if not self.initialize_data:
            raise ValueError(
                "Initialize data not calculated. Call calculate_parameters() first."
            )

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 清空并写入新内容（'w'模式会自动清空）
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.initialize_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            raise RuntimeError(f"保存初始化数据失败: {str(e)}") from e

    def initialize_model(self, grid_size=10.0, time_step=1.0):
        """执行完整的初始化流程

        Args:
            grid_size: 网格尺寸(mm)
            time_step: 时间步长(秒)

        Returns:
            Dict: 初始化数据

        Raises:
            ValueError: 如果网格尺寸或时间步长不合理
            RuntimeError: 如果计算超时或失败
        """
        # 参数校验
        if grid_size <= 0 or time_step <= 0:
            raise ValueError("网格尺寸和时间步长必须大于0")

        if grid_size > 50:  # 限制最大网格尺寸避免计算量过大
            raise ValueError("网格尺寸不能超过50mm")

        try:
            self.load_const_results()
            self.load_process_data()
            self.calculate_parameters(grid_size=grid_size, time_step=time_step)
            self.save_initialize_data()
            return self.initialize_data
        except Exception as e:
            raise RuntimeError(f"模型初始化失败: {str(e)}") from e


def initialize_model(grid_size=10.0, time_step=1.0):
    """提供给ui.py调用的快捷函数

    Args:
        grid_size: 网格尺寸(mm)
        time_step: 时间步长(秒)

    Returns:
        Dict: 初始化数据

    Raises:
        ValueError: 如果参数无效
        RuntimeError: 如果初始化失败
    """
    try:
        initializer = ModelInitializer()
        return initializer.initialize_model(grid_size=grid_size, time_step=time_step)
    except Exception as e:
        raise RuntimeError(f"初始化模型时出错: {str(e)}") from e
