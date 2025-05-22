import json
import pandas as pd
from Iterator import (
    solve_transient_heat_conduction,
    solve_transient_heat_conduction_with_phase_properties,
)
from pathlib import Path
import numpy as np


def process_segments():
    # 读取initialize.json
    with open("results/initialize.json", "r") as f:
        data = json.load(f)

    global_params = data["global_parameters"]
    segments = data["segments"]
    results = []

    for i, segment in enumerate(segments):
        # 直接使用global_params中的参数
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
            "Ts": global_params.get("Ts", 1450),  # 固相线温度 [K]
            "Tl": global_params.get("Tl", 1520),  # 液相线温度 [K]
            "props": global_params.get("props", {}),  # 钢种热物性字典
        }
        total_time1 = segment["time"]
        print(total_time1)
        print(f"Processing segment {i+1}/{len(segments)}")

        # 第一段使用initial_temp_field，后续使用前一段的结果
        if i == 0:
            initial_temp = np.array(segment["initial_temp_field"])
            if initial_temp.ndim != 2:
                raise ValueError("初始温度场必须是二维数组")
        else:
            initial_temp = results[-1]["final_temp"]
            if not isinstance(initial_temp, np.ndarray):
                raise ValueError("初始温度场必须是numpy数组")

        # 调用计算函数
        if params["property_method"] == "phase_based":
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

        # 保存结果到CSV
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
