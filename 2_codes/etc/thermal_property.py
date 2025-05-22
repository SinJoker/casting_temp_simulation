# 钢种物性参数数据

# 钢种物性参数数据
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


class ThermalProperties:
    """钢的热物性计算类"""

    def __init__(self, kind: str, Ts: float, Tl: float, Tc: float = 1800):
        """初始化热物性计算器
        Args:
            kind: 钢种名称
            Ts: 固相线温度(K)
            Tl: 液相线温度(K)
            Tc: 临界温度(K)，默认为1800
        """
        self.kind = kind
        self.Ts = Ts
        self.Tl = Tl
        self.Tc = Tc
        self.props = steel_properties.get(kind, {})

    def get_phase(self, T: float) -> str:
        """获取当前温度下的相态
        Returns:
            "solid" - 固相
            "mushy" - 两相区
            "liquid" - 液相
        """
        if T >= self.Tl:
            return "liquid"
        elif T >= self.Ts:
            return "mushy"
        return "solid"

    def lamda_cal(self, T: float, position: float) -> float:
        """计算等效热导率(W/m·K)
        Args:
            T: 当前温度(K)
            position: 位置参数(0-3)
        """
        lamdal = self.props["lamda_l"]
        lamdas = self.props["lamda_s"]
        fs = (self.Tl - T) / (self.Tl - self.Ts) if self.Tl != self.Ts else 0

        def _Acon(T: float, position: float) -> float:
            """计算Acon修正系数"""
            if 1 > position >= 0:  # 位置0-1区域
                if T >= self.Tl:  # 液相区
                    return 6 - 4 * ((self.Tc - T) / (self.Tc - self.Tl))
                elif self.Tl >= T >= self.Ts:  # 两相区
                    return 2 - 2 * fs
            elif 3 > position >= 1:  # 位置1-3区域
                if T >= self.Tl:  # 液相区
                    return 1 - ((self.Tc - T) / (self.Tc - self.Tl))
                elif self.Tl >= T >= self.Ts:  # 两相区
                    return 0
            return 0

        phase = self.get_phase(T)
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

    def cp_cal(self, T: float) -> float:
        """计算比热容(J/kg·K)
        Args:
            T: 当前温度(K)
        """
        phase = self.get_phase(T)
        if phase == "liquid":
            return self.props["c_l"]
        elif phase == "mushy":
            return self.props["c_m"]
        return self.props["c_s"]

    def rho_cal(self, T: float) -> float:
        """计算密度(kg/m3)
        Args:
            T: 当前温度(K)
        """
        phase = self.get_phase(T)
        if phase == "liquid":
            return self.props["rho_l"]
        elif phase == "mushy":
            return self.props["rho_m"]
        return self.props["rho_s"]


if __name__ == "__main__":
    # 使用示例
    # 初始化热物性计算器
    steel_kind = "中碳钢"
    Ts = 1400  # 固相线温度(K)
    Tl = 1500  # 液相线温度(K)
    thermal = ThermalProperties(steel_kind, Ts, Tl)

    # 计算不同温度下的物性
    temperatures = [1300, 1450, 1600]  # 固相区、两相区、液相区温度
    position = 0.5  # 位置参数

    print(f"钢种: {steel_kind}")
    print("温度(K) | 相态     | 热导率(W/m·K) | 比热容(J/kg·K) | 密度(kg/m3)")
    print("-" * 70)

    for T in temperatures:
        phase = thermal.get_phase(T)
        lamda = thermal.lamda_cal(T, position)
        cp = thermal.cp_cal(T)
        rho = thermal.rho_cal(T)
        print(f"{T:7} | {phase:8} | {lamda:14.2f} | {cp:14.2f} | {rho:10.2f}")
