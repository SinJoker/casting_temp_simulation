"""
calculation_processor.py

该模块负责处理温度场计算的主要逻辑，包括读取初始化参数、调用热传导求解器以及保存计算结果。
主要功能包括：
1. 读取 `initialize.json` 文件中的全局参数和分段参数。
2. 根据分段参数调用不同的热传导求解器（标准方法或相变基方法）。
3. 保存每段计算的结果到 CSV 文件，并将最后一段的温度场用于下一段的初始条件。

作者：孙俊博
日期：2025-05-28
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from Iterator import (
    solve_transient_heat_conduction,
    solve_transient_heat_conduction_with_phase_properties,
)


def process_segments():
    """
    处理温度场计算的分段逻辑。

    该函数读取 `initialize.json` 文件中的全局参数和分段参数，逐段调用热传导求解器，
    并保存每段的计算结果到 CSV 文件。最后一段的温度场将作为下一段的初始条件。

    返回:
        list: 包含每段计算结果的列表，每个结果是一个字典，包含最终温度场和输出路径。
    """
    # 读取 initialize.json 文件
    with open("2_codes/results/initialize.json", "r") as f:
        data = json.load(f)

    global_params = data["global_parameters"]
    segments = data["segments"]
    results = []

    for i, segment in enumerate(segments):
        # 构建当前段的参数
        params = {
            "Lx": global_params["Lx"],
            "Ly": global_params["Ly"],
            "dt": global_params["dt"],
            "tol": global_params["tol"],
            "T_inf_top": global_params.get("T_inf_top", 0.0),
            "T_inf_right": global_params.get("T_inf_right", 0.0),
            "boundary_type": global_params.get("boundary_type", 3),
            "h_top": global_params.get("h_top", 0.0),
            "h_right": global_params.get("h_right", 0.0),
            "sigma": global_params.get("sigma", 0.8),
            "property_method": global_params.get(
                "property_method", "phase_based"
            ),  # standard 或 phase_based
            "Ts": global_params.get("Ts", 1450),  # 固相线温度 [℃]
            "Tl": global_params.get("Tl", 1520),  # 液相线温度 [℃]
            "props": global_params.get("props", {}),  # 钢种热物性字典
        }
        total_time1 = segment["time"]
        print(params)
        print(total_time1)
        print(f"Processing segment {i+1}/{len(segments)}")

        # 设置初始温度场
        if i == 0:
            # 根据nx和ny创建初始温度场二维数组
            nx = global_params["nx"]
            ny = global_params["ny"]
            print(nx, ny)
            initial_temp = np.full((nx, ny), segment["initial_temp_field"], dtype=float)
        else:
            initial_temp = results[-1]["final_temp"]
            if not isinstance(initial_temp, np.ndarray):
                raise ValueError("初始温度场必须是numpy数组")

        print("Initial temperature field shape:", initial_temp.shape)

        # 根据 property_method 调用不同的求解器
        if params["property_method"] == "phase_based":
            # 调用相变基方法的热传导求解器
            X, Y, T_final, time_history, temp_history = (
                solve_transient_heat_conduction_with_phase_properties(
                    Lx=params["Lx"],
                    Ly=params["Ly"],
                    T_inf_top=params["T_inf_top"],
                    T_inf_right=params["T_inf_right"],
                    dt=params["dt"],
                    total_time=segment["time"],
                    initial_temp=initial_temp,
                    Ts=params["Ts"],
                    Tl=params["Tl"],
                    props=params["props"],
                    boundary_type=params["boundary_type"],
                    h_top=params["h_top"],
                    h_right=params["h_right"],
                    sigma=params["sigma"],
                    tol=params["tol"],
                )
            )
        else:
            # 调用标准方法的热传导求解器
            X, Y, T_final, time_history, temp_history = solve_transient_heat_conduction(
                Lx=params["Lx"],
                Ly=params["Ly"],
                T_inf_top=params["T_inf_top"],
                T_inf_right=params["T_inf_right"],
                dt=params["dt"],
                total_time=segment["time"],  # 修正为使用segment["time"]
                initial_temp=initial_temp,
                boundary_type=params["boundary_type"],
                h_top=params["h_top"],
                h_right=params["h_right"],
                sigma=params["sigma"],
                tol=params["tol"],
            )
        print(f"Segment {i+1} completed.")

        # 保存结果到CSV文件
        output_path = Path(f"2_codes/results/segment_{i+1}_results.csv")
        pd.DataFrame(temp_history).to_csv(output_path, index=False)

        # 保存最后时刻的温度分布用于下一段
        results.append(
            {
                "final_temp": T_final,  # 使用最终温度场
                "output_path": str(output_path),
            }
        )

    return results


if __name__ == "__main__":
    try:
        results = process_segments()
        print(f"计算完成！共处理了{len(results)}段数据")
        for i, result in enumerate(results):
            print(f"第{i+1}段结果保存至: {result['output_path']}")
    except Exception as e:
        print(f"计算出错: {str(e)}")
