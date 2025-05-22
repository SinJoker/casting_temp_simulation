"""钢的热物性计算函数"""


def get_conductivity(T):

    conductivity = (0.45041 - 1.7057 / 10000 * T) * 100

    return conductivity


def get_density(T):

    density = (7.8137 - 3.3481 / 10000 * T) * 1000

    return density


def get_specific_heat(T):
    """根据温度返回比热容(J/kg·K)
    使用高合金钢参数:
    - 固相(<1450℃): 658.811
    - 两相区(1450-1500℃): 650
    - 液相(>=1500℃): 691.667
    """
    if T < 1450:
        return 658.811
    elif 1450 <= T < 1500:
        return 650
    else:
        return 691.667


def get_phase(T: float, Ts: float, Tl: float) -> str:
    """获取当前温度下的相态
    Args:
        T: 当前温度(K)
        Ts: 固相线温度(K)
        Tl: 液相线温度(K)
    Returns:
        "solid" - 固相
        "mushy" - 两相区
        "liquid" - 液相
    """
    if T >= Tl:
        return "liquid"
    elif T >= Ts:
        return "mushy"
    return "solid"


def lamda_cal(
    T: float, position: float, Ts: float, Tl: float, Tc: float, props: dict
) -> float:
    """计算等效热导率(W/m·K)
    Args:
        T: 当前温度(K)
        position: 位置参数(0-3)
        Ts: 固相线温度(K)
        Tl: 液相线温度(K)
        Tc: 临界温度(K)
        props: 钢种热物性字典
    """
    lamdal = props["lamda_l"]
    lamdas = props["lamda_s"]
    fs = (Tl - T) / (Tl - Ts) if Tl != Ts else 0

    def _Acon(T: float, position: float) -> float:
        """计算Acon修正系数"""
        if 1 > position >= 0:  # 位置0-1区域
            if T >= Tl:  # 液相区
                return 6 - 4 * ((Tc - T) / (Tc - Tl))
            elif Tl >= T >= Ts:  # 两相区
                return 2 - 2 * fs
        elif 3 > position >= 1:  # 位置1-3区域
            if T >= Tl:  # 液相区
                return 1 - ((Tc - T) / (Tc - Tl))
            elif Tl >= T >= Ts:  # 两相区
                return 0
        return 0

    phase = get_phase(T, Ts, Tl)
    if phase == "liquid":
        return (1 + _Acon(T, position)) * lamdal
    elif phase == "mushy":
        # 并联模型
        lamdapar = (1 + _Acon(T, position)) * (1 - fs) * lamdal + fs * lamdas
        # 串联模型
        lamdaser = ((1 + _Acon(T, position)) * lamdal * lamdas) / (
            (1 + _Acon(T, position)) * lamdal * fs + (1 - fs) * lamdas
        )
        return (lamdapar + lamdaser) / 2
    return lamdas


def cp_cal(T: float, Ts: float, Tl: float, props: dict) -> float:
    """计算比热容(J/kg·K)
    Args:
        T: 当前温度(K)
        Ts: 固相线温度(K)
        Tl: 液相线温度(K)
        props: 钢种热物性字典
    """
    phase = get_phase(T, Ts, Tl)
    if phase == "liquid":
        return props["c_l"]
    elif phase == "mushy":
        return props["c_m"]
    return props["c_s"]


def rho_cal(T: float, Ts: float, Tl: float, props: dict) -> float:
    """计算密度(kg/m3)
    Args:
        T: 当前温度(K)
        Ts: 固相线温度(K)
        Tl: 液相线温度(K)
        props: 钢种热物性字典
    """
    phase = get_phase(T, Ts, Tl)
    if phase == "liquid":
        return props["rho_l"]
    elif phase == "mushy":
        return props["rho_m"]
    return props["rho_s"]
