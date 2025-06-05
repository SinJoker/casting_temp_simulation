import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class HeatTransferCalculator:
    """
    换热系数计算器类，用于计算结晶器区和二冷区的热流密度、换热系数等。
    该类提供了多种计算方法，支持不同场景下的热传递模拟。
    """

    def __init__(self):
        """
        初始化换热系数计算器。
        无需额外参数，直接实例化即可使用。
        """
        pass

    # 结晶器区热流边界条件
    def mold_heat_flux(self, t, A=2680, B=276):
        """
        计算结晶器热流密度。

        参数:
            t: 在结晶器内的时间(s)
            A: 公式系数，默认值为2680
            B: 公式系数，默认值为276

        返回:
            热流密度(kW/m²)
        """
        return A - B * np.sqrt(t)

    def mold_heat_flux_alternative(self, Q_L, delta_T, B, v_g, L_M, Z):
        """
        计算结晶器热流密度(替代公式)。

        参数:
            Q_L: 内弧铜板的通水量(m³/s)
            delta_T: 内弧铜板进出口水温差(℃)
            B: 板坯宽度(m)
            v_g: 拉坯速度(m/s)
            L_M: 结晶器有效长度(m)
            Z: 距离弯月面距离(m)

        返回:
            热流密度(kW/m²)
        """
        K1 = (3 * Q_L * delta_T * 4.18685) / (
            4 * B * (v_g**0.56) * (1 - np.exp(-1.5 * L_M))
        )
        return K1 * 2 * (v_g**0.56) * np.exp(-1.5 * Z) * 1e6

    # 二冷区对流换热边界条件
    def secondary_cooling_h(self, V, T_w, T_s=None, method="Mitsutsuka"):
        """
        计算二冷区换热系数。

        参数:
            V: 水量密度(L/m²·s)
            T_s: 板坯表面温度(℃)
            T_w: 冷却水温度(℃)
            method: 计算方法选择

        返回:
            换热系数(kW/m²·K)
        """
        V = max(V, 0.001)  # 避免零或负值

        if method == "Mitsutsuka":
            n = 0.7  # 0.65<n<0.75
            b = 0.0065  # 0.005<b<0.008
            h = (V**n) * (1 - b * T_w)

        elif method == "Shimada":
            h = 1.57 * (V**0.55) * (1 - 0.0075 * T_w)

        elif method == "Miikar_0.276":
            h = 0.0776 * V

        elif method == "Miikar_0.620":
            h = 0.1 * V

        elif method == "Ishiguro":
            h = 0.581 * (V**0.451) * (1 - 0.0075 * T_w)

        elif method == "Bolle_Moureou":
            if 1 < V < 7 and 627 < T_s < 927:
                h = 0.423 * (V**0.556)
            elif 0.8 < V < 2.5 and 727 < T_s < 1027:
                h = 0.360 * (V**0.556)
            else:
                h = 0

        elif method == "Sasaki":
            h_kcal = 708 * (V**0.75) * (T_s**-1.2) + 0.116
            h = h_kcal * 1.163 / 1000  # 转换为kW/m²·K

        elif method == "Mizikar":
            h = (0.076 - 0.10 * V) * 1.163 / 1000  # kcal/m²·h·℃ → kW/m²·K

        elif method == "Concast":
            h_kcal = 0.875 * 5748 * (1 - 0.0075 * T_w) * (V**0.451)
            h = h_kcal * 1.163 / 1000  # 转换为kW/m²·K

        elif method == "BUIST":
            if 4.5 < V < 20:
                h_kcal = 0.35 * V + 0.13
                h = h_kcal * 1.163 / 1000  # 转换为kW/m²·K
            else:
                h = 0

        elif method == "CaiKaike":
            h_kcal = 2.25e4 * (1 - 0.00075 * T_w) * (V**0.55)
            h = h_kcal * 1.163 / 1000  # 转换为kW/m²·K

        elif method == "ZhangKeqiang":
            if T_s == 900:
                h = 0.37 + 0.35 * (V**0.954)
            else:
                h = 0

        elif method == "Billet":
            if T_s > 900:
                h = 1.095e12 * (T_s**-4.15) * (V**0.75)
            elif 500 < T_s <= 900:
                h = 3.78e3 * (T_s**-1.34) * (V**0.785)
            else:
                h = 3.78e3 * (500**-1.34) * (V**0.785)

        elif method == "Sasaki_K":
            if 600 <= T_s <= 900:
                h_kcal = 2.293e8 * (V**0.616) * (T_s**-2.445)
                h = h_kcal * 1.163 / 1000
            elif 900 < T_s <= 1200:
                h_kcal = 2.830e7 * (V**0.75) * (T_s**-1.2)
                h = h_kcal * 1.163 / 1000
            else:
                h = 0

        elif method == "Concast_Journal":
            h_kcal = 9.0 * (0.276**0.2) * (V**0.75)
            h = h_kcal * 1.163 / 1000

        elif method == "Tegurashi":
            W_a = 10  # NL/m²·s
            K_T = 1.0
            h_kcal = 280.56 * (W_a**0.1373) * (V**0.75) * K_T
            h = h_kcal * 1.163 / 1000

        elif method == "Nippon_Steel":
            h_kcal = 9.0 * (V**0.85) + 100
            h = h_kcal * 1.163 / 1000

        elif method == "Okamura":
            v_a = 21.5  # m/s
            T_h = 293  # K
            h_rad = 5.67e-8 * 0.8 * ((T_s + 273) ** 4 - T_h**4) / (T_s + 273 - T_h)
            h = (5.35 * (T_s**0.12) * (V**0.52) * (v_a**0.37) + h_rad) / 1000

        elif method == "Kashima":
            v_a = 20  # m/s
            z = 1
            h_kcal = 10**1.48 * (V**0.6293) * (T_s**-0.1358) * (v_a**0.2734) * z
            h = h_kcal * 1.163 / 1000

        elif method == "Hitachi":
            h_kcal = 70.4 * (V**0.31343)
            h = h_kcal * 1.163 / 1000

        elif method == "Muller_Jeachar":
            uc = 21.5  # m/s
            h_kcal = 0.42 * (V**0.35) * (uc**0.5)
            h = h_kcal * 1.163 / 1000  # 转换为kW/m²·K

        else:
            raise ValueError(f"未知的计算方法: {method}")

        return max(h, 0)  # 确保非负

    def air_cooling_heat_flux(self, T_s, T_a, emissivity=0.8):
        """
        计算空冷区热流密度。

        参数:
            T_s (float或numpy.ndarray): 表面温度(℃)
            T_a (float或numpy.ndarray): 环境温度(℃)
            emissivity (float, optional): 发射率，默认为0.8

        返回:
            float或numpy.ndarray: 热流密度(W/m²)，当表面温度<=环境温度时返回0
        """
        import numpy as np

        if isinstance(T_s, np.ndarray) or isinstance(T_a, np.ndarray):
            # 处理数组输入
            T_s = np.asarray(T_s)
            T_a = np.asarray(T_a)
            q = np.zeros_like(T_s)
            mask = T_s > T_a
            q[mask] = 5.67e-8 * emissivity * ((T_s[mask] + 273) ** 4 - (T_a + 273) ** 4)
            return q
        else:
            # 处理标量输入
            if T_s <= T_a:
                return 0
            q = 5.67e-8 * emissivity * ((T_s + 273) ** 4 - (T_a + 273) ** 4)
            return q

    def plot_mold_heat_flux(self, t_max, a, b):
        """
        绘制结晶器热流密度随时间变化的曲线。

        参数:
            t_max: 最大时间(s)
            a: 公式系数A
            b: 公式系数B

        返回:
            plotly Figure对象
        """
        t = np.linspace(0, t_max, 50)
        q = a - b * np.sqrt(t)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=t,
                y=q,
                mode="lines",
                name="结晶器热流密度随时间变化的关系，单位：W/m²",
                line=dict(color="royalblue", width=2),
            )
        )
        fig.add_annotation(
            x=0.95,
            y=0.95,
            text=f"函数曲线: q = {a} - {b}√t",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=16, color="black"),
            align="right",
            xanchor="right",
            yanchor="top",
        )
        fig.update_layout(
            title=f"结晶器热流密度随时间变化的关系，单位：W/m²",
            xaxis_title="时间 s",
            yaxis_title="热流密度 W/m²",
            xaxis=dict(showgrid=True, gridcolor="lightgray"),
            yaxis=dict(showgrid=True, gridcolor="lightgray"),
            plot_bgcolor="white",
        )
        return fig

    def plot_cooling_water_distribution(
        self,
        mold_exit_time,
        last_zone_end_time,
        top_water_flows,
        side_water_flows,
        zone_names,
    ):
        """
        绘制二冷区水量分布直方图。

        参数:
            mold_exit_time (float): 结晶器出口时间(s)
            last_zone_end_time (float): 最后一个二冷区结束时间(s)
            top_water_flows (list): 各二冷区顶面水量(L/m²·s)
            side_water_flows (list): 各二冷区侧面水量(L/m²·s)
            zone_names (list): 各二冷区名称

        返回:
            plotly Figure对象
        """
        zone_centers = np.linspace(
            mold_exit_time, last_zone_end_time, len(zone_names) + 1
        )
        zone_centers = (zone_centers[:-1] + zone_centers[1:]) / 2

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("顶面水量分布", "侧面水量分布"),
            shared_xaxes=True,
            vertical_spacing=0.1,
        )

        fig.add_trace(
            go.Bar(
                x=zone_centers,
                y=top_water_flows,
                name="顶面水量",
                marker_color="royalblue",
                text=zone_names,
                textposition="auto",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=zone_centers,
                y=side_water_flows,
                name="侧面水量",
                marker_color="firebrick",
                text=zone_names,
                textposition="auto",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=600,
            showlegend=False,
            font=dict(family="SimHei", size=12),
            title_text="二冷区水量分布",
            bargap=0.2,
        )
        fig.update_xaxes(title_text="时间 (s)", row=2, col=1)
        fig.update_yaxes(title_text="顶面水量 (L/m²·s)", row=1, col=1)
        fig.update_yaxes(title_text="侧面水量 (L/m²·s)", row=2, col=1)

        return fig

    def plot_heat_transfer_coefficients(
        self,
        mold_exit_time,
        last_zone_end_time,
        h_values,
        zone_names,
        casting_speed=1.2,
    ):
        """
        绘制换热系数分布直方图。

        参数:
            mold_exit_time: 结晶器出口时间(s)
            last_zone_end_time: 最后一个二冷区结束时间(s)
            h_values: 各分区换热系数列表[{"top": h_top, "side": h_side}]
            zone_names: 各分区名称列表
            casting_speed: 拉速(m/min)

        返回:
            plotly Figure对象
        """
        casting_speed_mm_s = casting_speed * 1000 / 60
        zone_times = [mold_exit_time]
        for i in range(len(zone_names)):
            zone_length = last_zone_end_time[i]
            zone_duration = zone_length / casting_speed_mm_s
            zone_times.append(zone_times[-1] + zone_duration)

        zone_centers = [
            (zone_times[i] + zone_times[i + 1]) / 2 for i in range(len(zone_times) - 1)
        ]

        h_top = [h["top"] for h in h_values]
        h_side = [h["side"] for h in h_values]

        hover_text_top = [
            f"分区: {zone_names[i]}<br>"
            f"时间范围: {zone_times[i]:.2f}-{zone_times[i+1]:.2f}s<br>"
            f"换热系数: {h_top[i]:.2f} kW/m²·K"
            for i in range(len(zone_names))
        ]

        hover_text_side = [
            f"分区: {zone_names[i]}<br>"
            f"时间范围: {zone_times[i]:.2f}-{zone_times[i+1]:.2f}s<br>"
            f"换热系数: {h_side[i]:.2f} kW/m²·K"
            for i in range(len(zone_names))
        ]

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("顶面换热系数分布", "侧面换热系数分布"),
            shared_xaxes=True,
            vertical_spacing=0.1,
        )

        fig.add_trace(
            go.Bar(
                x=zone_centers,
                y=h_top,
                name="顶面换热系数",
                marker_color="royalblue",
                text=zone_names,
                textposition="auto",
                hovertext=hover_text_top,
                hoverinfo="text",
                width=[
                    zone_times[i + 1] - zone_times[i]
                    for i in range(len(zone_times) - 1)
                ],
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=zone_centers,
                y=h_side,
                name="侧面换热系数",
                marker_color="firebrick",
                text=zone_names,
                textposition="auto",
                hovertext=hover_text_side,
                hoverinfo="text",
                width=[
                    zone_times[i + 1] - zone_times[i]
                    for i in range(len(zone_times) - 1)
                ],
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=600,
            showlegend=False,
            font=dict(family="SimHei", size=12),
            title_text="二冷区换热系数分布",
            bargap=0.2,
        )
        fig.update_xaxes(title_text="时间 (s)", row=2, col=1)
        fig.update_yaxes(title_text="顶面换热系数 (kW/m²·K)", row=1, col=1)
        fig.update_yaxes(title_text="侧面换热系数 (kW/m²·K)", row=2, col=1)

        return fig
