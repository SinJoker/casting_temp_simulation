import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class SteelProperty:
    """钢种物性数据类"""

    lamda_s: float  # 固相热导率 (W/m·K)
    lamda_m: float  # 两相区热导率 (W/m·K)
    lamda_l: float  # 液相热导率 (W/m·K)
    c_s: float  # 固相比热容 (J/kg·K)
    c_m: float  # 两相区比热容 (J/kg·K)
    c_l: float  # 液相比热容 (J/kg·K)
    rho_s: float  # 固相密度 (kg/m3)
    rho_m: float  # 两相区密度 (kg/m3)
    rho_l: float  # 液相密度 (kg/m3)
    l_f: float  # 潜热 (J/kg)


def load_steel_properties(file_path: Optional[str] = None) -> Dict[str, SteelProperty]:
    """从JSON文件加载钢种物性参数

    Args:
        file_path: JSON文件路径，默认为同目录下的steel_properties.json

    Returns:
        钢种物性参数字典

    Raises:
        FileNotFoundError: 当JSON文件不存在时
        json.JSONDecodeError: 当JSON文件格式错误时
        ValueError: 当数据验证失败时
    """
    if file_path is None:
        file_path = str(Path(__file__).parent / "steel_properties.json")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 数据验证和转换
    steel_props = {}
    for steel_type, props in data.items():
        try:
            steel_props[steel_type] = SteelProperty(**props)
        except TypeError as e:
            raise ValueError(f"钢种 '{steel_type}' 属性不完整或类型错误: {str(e)}")

    return steel_props


# 加载钢种物性参数
try:
    steel_properties = load_steel_properties()
except Exception as e:
    # 如果加载失败，使用内置默认值
    steel_properties = {
        "高合金钢": {
            "lamda_s": 29.008,
            "lamda_m": 35.470,
            "lamda_l": 35.470,
            "c_s": 658.811,
            "c_m": 650,
            "c_l": 691.667,
            "rho_s": 7600,
            "rho_m": 7400,
            "rho_l": 7200,
            "l_f": 270000,
        },
        "低合金钢": {
            "lamda_s": 31.333,
            "lamda_m": 41.270,
            "lamda_l": 41.270,
            "c_s": 665.311,
            "c_m": 700,
            "c_l": 743.5,
            "rho_s": 7600,
            "rho_m": 7400,
            "rho_l": 7200,
            "l_f": 270000,
        },
        "中合金钢": {
            "lamda_s": 30.672,
            "lamda_m": 39.955,
            "lamda_l": 39.955,
            "c_s": 661.975,
            "c_m": 700,
            "c_l": 740.556,
            "rho_s": 7600,
            "rho_m": 7400,
            "rho_l": 7200,
            "l_f": 274950,
        },
        "包晶合金钢": {
            "lamda_s": 30.667,
            "lamda_m": 39.075,
            "lamda_l": 39.075,
            "c_s": 664.083,
            "c_m": 700,
            "c_l": 753.571,
            "rho_s": 7600,
            "rho_m": 7400,
            "rho_l": 7200,
            "l_f": 270000,
        },
        "高碳钢": {
            "lamda_s": 29.008,
            "lamda_m": 35.470,
            "lamda_l": 35.470,
            "c_s": 658.811,
            "c_m": 650,
            "c_l": 691.667,
            "rho_s": 7600,
            "rho_m": 7400,
            "rho_l": 7200,
            "l_f": 270000,
        },
        "低碳钢": {
            "lamda_s": 31.333,
            "lamda_m": 41.270,
            "lamda_l": 41.270,
            "c_s": 665.311,
            "c_m": 700,
            "c_l": 743.5,
            "rho_s": 7600,
            "rho_m": 7400,
            "rho_l": 7200,
            "l_f": 270000,
        },
        "中碳钢": {
            "lamda_s": 30.672,
            "lamda_m": 39.955,
            "lamda_l": 39.955,
            "c_s": 661.975,
            "c_m": 700,
            "c_l": 740.556,
            "rho_s": 7600,
            "rho_m": 7400,
            "rho_l": 7200,
            "l_f": 274950,
        },
        "包晶钢": {
            "lamda_s": 31.041,
            "lamda_m": 41.700,
            "lamda_l": 41.700,
            "c_s": 663.641,
            "c_m": 700,
            "c_l": 740,
            "rho_s": 7600,
            "rho_m": 7400,
            "rho_l": 7200,
            "l_f": 270000,
        },
    }


def calculate_liquidus_temp(formula_name: str, composition: dict) -> float:
    """计算指定液相线公式的温度值"""

    def get_value(key):
        return composition.get(key, 0.0)

    C = get_value("C")
    Si = get_value("Si")
    Mn = get_value("Mn")
    P = get_value("P")
    S = get_value("S")
    Ni = get_value("Ni")
    Cr = get_value("Cr")
    Mo = get_value("Mo")
    Cu = get_value("Cu")
    V = get_value("V")
    W = get_value("W")
    N = get_value("N")
    Al = get_value("Al")
    Co = get_value("Co")
    Ti = get_value("Ti")
    O = get_value("O")
    Nb = get_value("Nb")
    Ta = get_value("Ta")
    As = get_value("As")
    Sn = get_value("Sn")
    Zr = get_value("Zr")

    try:
        if formula_name == "1980_国外连铸新技术":
            return 1536 - (
                90 * C
                + 6 * Si
                + 1.7 * Mn
                + 28 * P
                + 40 * S
                + 2.6 * Cu
                + 2.9 * Ni
                + 1.8 * Cr
                + 5.1 * Al
            )

        elif formula_name == "1988_商家K":
            return 1536 - (
                78 * C
                + 7.6 * Si
                + 4.9 * Mn
                + 34 * P
                + 30 * S
                + 5 * Cu
                + 3.1 * Ni
                + 1.3 * Cr
                + 3.6 * Al
            )

        elif formula_name == "1989_日本广田连铸技术":
            return 1536 - (
                90 * C
                + 6 * Si
                + 1.7 * Mn
                + 28 * P
                + 40 * S
                + 2.6 * Cu
                + 2.9 * Ni
                + 1.8 * Cr
                + 5.1 * Al
            )

        elif formula_name == "1990_连续铸钢手册_通用":
            return 1536 - (
                90 * C
                + 6 * Si
                + 1.7 * Mn
                + 28 * P
                + 40 * S
                + 2.6 * Cu
                + 2.9 * Ni
                + 1.8 * Cr
                + 5.1 * Al
            )

        elif formula_name == "1990_连续铸钢手册_碳素钢":
            return 1536 - (
                90 * C
                + 6 * Si
                + 1.7 * Mn
                + 28 * P
                + 40 * S
                + 2.6 * Cu
                + 2.9 * Ni
                + 1.8 * Cr
                + 5.1 * Al
            )

        elif formula_name == "1990_连续铸钢手册_特殊钢":
            return 1536 - (
                78 * C
                + 7.6 * Si
                + 4.9 * Mn
                + 34 * P
                + 30 * S
                + 5 * Cu
                + 3.1 * Ni
                + 1.3 * Cr
                + 3.6 * Al
            )

        elif formula_name == "1994_连续铸钢原理与工艺":
            return 1536 - (
                90 * C
                + 6 * Si
                + 1.7 * Mn
                + 28 * P
                + 40 * S
                + 2.6 * Cu
                + 2.9 * Ni
                + 1.8 * Cr
                + 5.1 * Al
            )

        elif formula_name == "1997_宝钢技术_平居公式":
            return 1536 - (
                90 * C
                + 6 * Si
                + 1.7 * Mn
                + 28 * P
                + 40 * S
                + 2.6 * Cu
                + 2.9 * Ni
                + 1.8 * Cr
                + 5.1 * Al
            )

        elif formula_name == "2000_商家D_不锈钢1":
            return 1536 - (
                78 * C
                + 7.6 * Si
                + 4.9 * Mn
                + 34 * P
                + 30 * S
                + 5 * Cu
                + 3.1 * Ni
                + 1.3 * Cr
                + 3.6 * Al
            )

        elif formula_name == "2000_商家D_不锈钢2":
            return 1536 - (
                78 * C
                + 7.6 * Si
                + 4.9 * Mn
                + 34 * P
                + 30 * S
                + 5 * Cu
                + 3.1 * Ni
                + 1.3 * Cr
                + 3.6 * Al
            )

        elif formula_name == "2006_薄板坯连铸连轧":
            return 1536 - (
                90 * C
                + 6 * Si
                + 1.7 * Mn
                + 28 * P
                + 40 * S
                + 2.6 * Cu
                + 2.9 * Ni
                + 1.8 * Cr
                + 5.1 * Al
            )

        elif formula_name == "2013_炼钢_通用":
            return 1536 - (
                90 * C
                + 6 * Si
                + 1.7 * Mn
                + 28 * P
                + 40 * S
                + 2.6 * Cu
                + 2.9 * Ni
                + 1.8 * Cr
                + 5.1 * Al
            )

        elif formula_name == "1998_商家A_镀锡板":
            return 1536 - (
                90 * C
                + 6 * Si
                + 1.7 * Mn
                + 28 * P
                + 40 * S
                + 2.6 * Cu
                + 2.9 * Ni
                + 1.8 * Cr
                + 5.1 * Al
            )

        elif formula_name == "2000_商家B_不锈钢":
            return 1536 - (
                78 * C
                + 7.6 * Si
                + 4.9 * Mn
                + 34 * P
                + 30 * S
                + 5 * Cu
                + 3.1 * Ni
                + 1.3 * Cr
                + 3.6 * Al
            )

        elif formula_name == "2003_武钢_硅钢":
            return 1536 - (
                90 * C
                + 6 * Si
                + 1.7 * Mn
                + 28 * P
                + 40 * S
                + 2.6 * Cu
                + 2.9 * Ni
                + 1.8 * Cr
                + 5.1 * Al
            )

        elif formula_name == "2007_连铸900问":
            return 1536 - (
                90 * C
                + 6 * Si
                + 1.7 * Mn
                + 28 * P
                + 40 * S
                + 2.6 * Cu
                + 2.9 * Ni
                + 1.8 * Cr
                + 5.1 * Al
            )

        elif formula_name == "商家D_碳钢分段":
            if C <= 0.1:
                return 1536 - (90 * C + 6 * Si + 1.7 * Mn + 28 * P + 40 * S)
            elif 0.1 < C <= 0.2:
                return 1536 - (80 * C + 6 * Si + 1.7 * Mn + 28 * P + 40 * S)
            else:
                return 1536 - (70 * C + 6 * Si + 1.7 * Mn + 28 * P + 40 * S)

        elif formula_name == "铸铁":
            return 1536 - (
                90 * C
                + 6 * Si
                + 1.7 * Mn
                + 28 * P
                + 40 * S
                + 2.6 * Cu
                + 2.9 * Ni
                + 1.8 * Cr
                + 5.1 * Al
            )

        else:
            print(f"[WARNING] 未知液相线公式: {formula_name}")
            return 1500.0  # 默认值

    except Exception as e:
        print(f"[ERROR] 计算液相线温度时出错({formula_name}): {str(e)}")
        return 1500.0  # 默认值


def calculate_solidus_temp(formula_name: str, composition: dict) -> float:
    """计算指定固相线公式的温度值"""

    def get_value(key):
        return composition.get(key, 0.0)

    C = get_value("C")
    Si = get_value("Si")
    Mn = get_value("Mn")
    P = get_value("P")
    S = get_value("S")
    Ni = get_value("Ni")
    Cr = get_value("Cr")
    Mo = get_value("Mo")
    Cu = get_value("Cu")
    V = get_value("V")
    W = get_value("W")
    N = get_value("N")
    Al = get_value("Al")
    Co = get_value("Co")
    Ti = get_value("Ti")
    O = get_value("O")
    Nb = get_value("Nb")
    Ta = get_value("Ta")
    As = get_value("As")
    Sn = get_value("Sn")
    Zr = get_value("Zr")

    try:
        if formula_name == "1997_宝钢技术_平居公式":
            base = (
                20.5 * Si
                + 6.5 * Mn
                + 2.0 * Cr
                + 11.5 * Ni
                + 5.5 * Al
                + 500 * P
                + 700 * S
            )
            if C <= 0.09:
                return 1538 - (478 * C + base)
            elif 0.09 < C <= 0.17:
                return 1495 - base
            else:
                return 1527 - (187.5 * C + base)

        elif formula_name == "2006_薄板坯连铸连轧":
            return 1537 - (
                175 * C + 20 * Si + 30 * Mn + 280 * P + 575 * S + 7.5 * Al + 160 * O
            )

        elif formula_name == "2007_连铸900问":
            return 1534 - 2.29 * (
                80.5 * C
                + 17.8 * Si
                + 3.75 * Mn
                + 33.5 * P
                + 33.5 * S
                + 3 * N
                + 1.5 * Cr
                + 3.4 * Cu
                + 3.4 * Al
            )

        elif formula_name == "2011_连续铸钢生产技术":
            return 1471 - (
                25.2 * C
                + 12 * Si
                + 7.6 * Mn
                + 34 * P
                + 30 * S
                + 5 * Cu
                + 3.1 * Ni
                + 1.3 * Cr
                + 3.6 * Al
                + 2 * Mo
                + 2 * V
                + 18 * Ti
            )

        elif formula_name == "2012_钢铁":
            return 1536 - (
                175 * C
                + 20 * Si
                + 30 * Mn
                + 280 * P
                + 575 * S
                + 6.5 * Cr
                + 4 * V
                + 4.75 * Ni
                + 7.5 * Al
                + 2.5 * W
                + 40 * Ti
                + 5 * Mo
                + 60 * Nb
                + 160 * O
            )

        else:
            print(f"[WARNING] 未知固相线公式: {formula_name}")
            return 1400.0  # 默认值

    except Exception as e:
        print(f"[ERROR] 计算固相线温度时出错({formula_name}): {str(e)}")
        return 1400.0  # 默认值


def calculate_const_properties(kind: str) -> dict:
    """计算指定钢种的常温物性参数

    Args:
        kind: 钢种名称

    Returns:
        该钢种的物性字典

    Raises:
        ValueError: 当钢种不存在时
    """
    if kind not in steel_properties:
        raise ValueError(f"无效的钢种类型: {kind}")

    props = steel_properties[kind]
    if isinstance(props, SteelProperty):
        return {
            "lamda_s": props.lamda_s,
            "lamda_m": props.lamda_m,
            "lamda_l": props.lamda_l,
            "c_s": props.c_s,
            "c_m": props.c_m,
            "c_l": props.c_l,
            "rho_s": props.rho_s,
            "rho_m": props.rho_m,
            "rho_l": props.rho_l,
            "l_f": props.l_f,
        }
    else:  # 处理字典格式
        return {
            "lamda_s": props["lamda_s"],
            "lamda_m": props["lamda_m"],
            "lamda_l": props["lamda_l"],
            "c_s": props["c_s"],
            "c_m": props["c_m"],
            "c_l": props["c_l"],
            "rho_s": props["rho_s"],
            "rho_m": props["rho_m"],
            "rho_l": props["rho_l"],
            "l_f": props["l_f"],
        }
