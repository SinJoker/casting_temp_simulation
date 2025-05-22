# 必须在最前面调用set_page_config
import streamlit as st
from sympy import symbols, sympify, latex

# 定义公式中可能用到的符号变量
V, T_w, h, T, T_s, T_l, T_m, q, k, rho, c_p = symbols(
    "V T_w h T T_s T_l T_m q k rho c_p"
)
import pandas as pd
import json
import os
from thermal_properties import (
    calculate_const_properties,
    calculate_liquidus_temp,
    calculate_solidus_temp,
)
from model_initializer import initialize_model
from prop_vs_temp import (
    cp_cal,
    lamda_cal,
    rho_cal,
)

from boundary_condition import HeatTransferCalculator

# 绘制物性参数图表(使用plotly)
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

# """设置连铸区温度场模拟的UI界面"""
st.title("连铸区温度场模拟计算")

# 初始化session state
if "components" not in st.session_state:
    st.session_state.components = []

tab1, tab2, tab3 = st.tabs(["1️⃣ 钢物性参数", "2️⃣ 工艺及设备参数", " 3️⃣ 计算参数"])

with tab1:
    # 从json文件读取钢种物性参数和预设元素
    json_path = os.path.join(
        os.path.dirname(__file__), "resources", "steel_properties.json"
    )
    with open(json_path, "r", encoding="utf-8") as f:
        steel_data = json.load(f)
        steel_properties = {
            k: v
            for k, v in steel_data.items()
            if k not in ["preset_elements", "all_components"]
        }
        preset_elements = steel_data["preset_elements"]
        all_components = steel_data["all_components"]
    col1, col2 = st.columns([3, 7], gap="small", border=True)

    with col1:
        st.header("钢物性参数设置")

        def on_steel_type_change():
            # 清空现有成分
            st.session_state.components = []
            current_steel = st.session_state.steel_type
            # st.write(f"切换钢种到: {current_steel}")

            # 重新加载新钢种的预设成分
            if current_steel in preset_elements:
                # st.write(f"找到预设元素: {preset_elements[current_steel]}")
                for elem in all_components:
                    if elem in preset_elements[current_steel]:
                        value = preset_elements[current_steel][elem]
                        st.session_state.components.append(
                            {"name": elem, "percentage": value}
                        )
                        # st.write(f"添加元素: {elem} = {value}%")

                # 标记需要刷新(仅当不在刷新过程中)
                if not st.session_state.get("refreshing", False):
                    st.session_state.need_refresh = True

        spec = st.selectbox(
            "选择一个钢种：",
            list(preset_elements.keys()),
            key="steel_type",
            on_change=on_steel_type_change,
        )

        # 初始化元素百分比字典
        element_percentages = {elem: 0.0 for elem in all_components}

        # 检查是否需要刷新(避免重复触发)
        if st.session_state.get("need_refresh", False) and not st.session_state.get(
            "refreshing", False
        ):
            st.session_state.refreshing = True
            st.session_state.need_refresh = False
            st.rerun()

        # 重置刷新状态
        if st.session_state.get("refreshing", False):
            st.session_state.refreshing = False

        # 初始化时加载预设成分
        if "components" not in st.session_state or not st.session_state.components:
            if spec in preset_elements:
                st.session_state.components = []
                # st.write(f"初始化加载钢种: {spec}")
                # st.write(f"预设元素: {preset_elements[spec]}")

                for elem in all_components:
                    if elem in preset_elements[spec]:
                        value = preset_elements[spec][elem]
                        element_percentages[elem] = value
                        st.session_state.components.append(
                            {"name": elem, "percentage": value}
                        )
                        # st.write(f"初始化添加元素: {elem} = {value}%")

        # 使用从JSON文件读取的元素列表

        # 初始化元素百分比字典
        element_percentages = {elem: 0.0 for elem in all_components}

        # 设置预设元素的百分比并添加到selected_components
        if spec in preset_elements:
            # 按照all_components顺序添加元素
            for elem in all_components:
                if elem in preset_elements[spec]:
                    value = preset_elements[spec][elem]
                    element_percentages[elem] = value
                    # 如果元素不在已选列表中，则添加
                    if not any(
                        c["name"] == elem
                        for c in st.session_state.get("components", [])
                    ):
                        new_component = {"name": elem, "percentage": value}
                        if "components" not in st.session_state:
                            st.session_state.components = []
                        st.session_state.components.append(new_component)

        # 允许所有选项添加自定义成分
        selected_components = st.session_state.get("components", [])

        # 更新可用组件列表
        used_names = [comp["name"] for comp in selected_components]
        available_components = [c for c in all_components if c not in used_names]

        tab1_col1, tab1_col2 = col1.columns([1, 1], gap="small")
        # 成分管理逻辑
        if (
            tab1_col1.button("添加成分", use_container_width=True)
            and available_components
        ):
            # 添加第一个可用元素
            elem = next((e for e in all_components if e in available_components), None)
            if elem:
                selected_components.append({"name": elem, "percentage": 0.0})
                st.session_state.components = selected_components
                st.rerun()

        # 显示和编辑现有成分
        for i, comp in enumerate(selected_components):
            with st.container(border=True):
                cols = st.columns([1, 1, 1], gap="small", vertical_alignment="bottom")

                # 成分名称选择
                with cols[0]:
                    current_name = comp["name"]
                    if current_name not in available_components:
                        available_components.insert(0, current_name)

                    new_name = st.selectbox(
                        f"成分名称 {i+1}",
                        available_components,
                        index=available_components.index(current_name),
                        key=f"name_select_{i}",
                    )
                    if new_name != current_name:
                        comp["name"] = new_name
                        st.rerun()

                # 百分比输入
                with cols[1]:
                    comp["percentage"] = st.number_input(
                        f"百分比 {i+1} %",
                        min_value=0.0,
                        value=comp["percentage"],
                        step=0.01,
                        key=f"percent_input_{i}",
                    )
                    element_percentages[new_name] = comp["percentage"]

                # 删除按钮
                with cols[2]:
                    if st.button(
                        "🗑️ 删除成分",  # 使用图标代替文字
                        key=f"delete_btn_{i}",
                        help=f"删除成分 {comp['name']}",
                        use_container_width=True,
                    ):
                        selected_components.pop(i)
                        st.session_state.components = selected_components
                        st.rerun()

        # 根据钢种设置默认index
        default_index = 5 if spec == "奥氏体不锈钢(不锈钢304/316)" else 0
        kind = st.selectbox(
            "钢的分类",
            [
                "低碳钢",
                "中碳钢",
                "高碳钢",
                "低合金钢",
                "中合金钢",
                "高合金钢",
                "包晶钢",
                "包晶合金钢",
            ],
            index=default_index,
            key="steel_kind_select",
        )
        # 从文件resources/formula_names.json读取液相线和固相线计算公式列表
        formula_path = os.path.join(
            os.path.dirname(__file__), "resources", "formula_names.json"
        )
        with open(formula_path, "r", encoding="utf-8") as f:
            formula_data = json.load(f)
            liquidus_formulas = formula_data["liquidus_formulas"]
            solidus_formulas = formula_data["solidus_formulas"]

        liquid_formula = st.selectbox("液相线的计算公式", liquidus_formulas)
        solid_formula = st.selectbox("固相线的计算公式", solidus_formulas)

    with col2:

        if (
            tab1_col2.button("保存并更新成分", use_container_width=True, type="primary")
            and selected_components
        ):
            # 构建一个json字段，用来储存元素成分，钢的分类，选用的液相线公式和固相线公式，
            basic_data = {
                "composition": {
                    comp["name"]: comp["percentage"] for comp in selected_components
                },
                "kind": kind,
                "liquidus_formula": liquid_formula,
                "solidus_formula": solid_formula,
            }

            # 将基础数据写入results/basic_data.json
            os.makedirs("results", exist_ok=True)
            with open("results/basic_data.json", "w", encoding="utf-8") as f:
                json.dump(basic_data, f, ensure_ascii=False, indent=4)

            # 调用compute_temperatures函数计算一些结果，并输出
            const_results = {
                "const_properties": calculate_const_properties(basic_data["kind"]),
                "liquid_temp": calculate_liquidus_temp(
                    basic_data["liquidus_formula"], basic_data["composition"]
                ),
                "solid_temp": calculate_solidus_temp(
                    basic_data["solidus_formula"], basic_data["composition"]
                ),
            }

            with open("results/const_results.json", "w", encoding="utf-8") as f:
                json.dump(const_results, f, ensure_ascii=False, indent=4)

            # 显示温度结果
            st.write("### 温度计算结果")
            cols = st.columns(2)
            with cols[0]:
                st.metric("液相线温度", f"{const_results['liquid_temp']:.2f} °C")
            with cols[1]:
                st.metric("固相线温度", f"{const_results['solid_temp']:.2f} °C")

            # 显示物性参数表格
            st.write("### 物性参数")
            const_props = const_results["const_properties"]
            # 参数名称翻译和后缀解释
            param_trans = {
                "lamda": "导热系数",
                "c": "比热容",
                "rho": "密度",
                "l_f": "潜热",
            }
            phase_trans = {"_s": "(固相)", "_m": "(两相)", "_l": "(液相)"}

            # 根据参数名称自动分配单位和中文名称
            unit_map = {
                "lamda": "W/(m·K)",
                "c": "J/(kg·K)",
                "rho": "kg/m³",
                "l_f": "kJ/kg",
            }

            # 创建带单位的表格
            prop_data = []
            for name, value in const_props.items():
                # 获取基础参数名和单位
                base_name = next((k for k in param_trans if name.startswith(k)), "")
                unit = unit_map.get(base_name, "")

                # 构建中文参数名
                chinese_name = param_trans.get(base_name, name)
                for suffix, phase in phase_trans.items():
                    if name.endswith(suffix):
                        chinese_name += phase
                        break

                prop_data.append(
                    {"参数名称": chinese_name, "参数值": value, "单位": unit}
                )

            prop_df = pd.DataFrame(prop_data)
            st.dataframe(prop_df, hide_index=True, use_container_width=True)

            # 获取物性参数
            props = const_results["const_properties"]
            Tl = const_results["liquid_temp"]
            Ts = const_results["solid_temp"]
            Tc = Tl + 100  # 假设临界温度比液相线高100℃

            # 创建温度范围(1600~1000℃)
            temps = np.linspace(1600, 1300, 50)
            # 创建距离范围(0-4m)
            positions = np.linspace(0, 4, 50)

            # 计算比热容和密度
            cps = [cp_cal(T, Ts, Tl, props) for T in temps]
            rhos = [rho_cal(T, Ts, Tl, props) for T in temps]

            # 计算导热系数(3D)
            T_grid, P_grid = np.meshgrid(temps, positions)
            lamdas = np.array(
                [[lamda_cal(T, p, Ts, Tl, Tc, props) for T in temps] for p in positions]
            )

            # 创建2x2网格布局
            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[
                    [{"type": "xy"}, {"type": "surface", "rowspan": 2}],
                    [{"type": "xy"}, None],
                ],
                subplot_titles=(
                    "密度随温度变化",
                    "导热系数随温度和距离变化",
                    "比热容随温度变化",
                ),
                vertical_spacing=0.1,
                horizontal_spacing=0.05,
            )

            # 左上: 密度图
            fig.add_trace(
                go.Scatter(
                    x=temps, y=rhos, name="密度 (kg/m³)", line=dict(color="blue")
                ),
                row=1,
                col=1,
            )
            fig.update_xaxes(title_text="温度 (℃)", row=1, col=1, range=[1600, 1300])
            fig.update_yaxes(title_text="密度 (kg/m³)", row=1, col=1)

            # 左下: 比热容图
            fig.add_trace(
                go.Scatter(
                    x=temps, y=cps, name="比热容 (J/kg·K)", line=dict(color="red")
                ),
                row=2,
                col=1,
            )
            fig.update_xaxes(title_text="温度 (℃)", row=2, col=1, range=[1600, 1300])
            fig.update_yaxes(title_text="比热容 (J/kg·K)", row=2, col=1)

            # 右边: 导热系数3D图 (跨两行)
            fig.add_trace(
                go.Surface(
                    x=T_grid,
                    y=P_grid,
                    z=lamdas,
                    name="导热系数",
                    colorscale="Viridis",
                    showscale=False,
                    contours_z=dict(
                        show=True,
                        usecolormap=True,
                        highlightcolor="limegreen",
                        project_z=True,
                    ),
                ),
                row=1,
                col=2,
            )
            # Plotly 支持的常见 colorscale 包括:

            # 连续色标:
            # Viridis (默认)
            # Plasma
            # Inferno
            # Magma
            # Cividis
            # Hot
            # Jet
            # Greys
            # YlGnBu
            # Greens
            # YlOrRd
            # Bluered
            # RdBu
            # Picnic
            # Rainbow
            # Portland
            # Electric
            # Blackbody
            # Earth
            # Thermal
            # 离散色标:
            # Blues
            # Reds
            # Greens
            # Purples
            # Oranges
            # BuPu
            # PuBu
            # PuRd
            # RdPu
            # BuGn
            # GnBu
            # PuBuGn
            # YlGn
            # YlOrBr
            # 特殊色标:
            # Turbo (类似 Jet 但更均匀)
            # HSV (色相-饱和度-明度)
            # Plotly3 (Plotly 默认色标)
            # 在温度场模拟中，常用的 colorscale 包括:

            # Hot/Thermal - 适合温度可视化
            # Viridis/Plasma - 科学可视化标准
            # Jet - 传统温度图
            # RdBu - 红蓝对比适合温差显示
            # 3D图视角和主题设置
            scene_settings = {
                "default": {
                    "camera": dict(eye=dict(x=-0.9, y=0.9, z=0.6)),  # 顺时针旋转90度
                    "bgcolor": "white",
                    "colorscale": "Viridis",
                },
                "dark": {
                    "camera": dict(eye=dict(x=-0.9, y=0.9, z=0.6)),
                    "bgcolor": "rgb(90,90,90)",
                    "colorscale": "Plasma",
                },
                "blue": {
                    "camera": dict(eye=dict(x=-0.9, y=0.9, z=0.6)),
                    "bgcolor": "rgb(240,248,255)",
                    "colorscale": "Blues",
                },
                "warm": {
                    "camera": dict(eye=dict(x=-0.9, y=0.9, z=0.6)),
                    "bgcolor": "white",
                    "colorscale": "Plasma",
                },
            }

            # 默认使用第一种主题
            selected_theme = "warm"
            fig.update_scenes(
                xaxis_title="温度 (℃)",
                yaxis_title="与弯月面距离 (m)",
                zaxis_title="导热系数 (W/m·K)",
                camera=scene_settings[selected_theme]["camera"],
                bgcolor=scene_settings[selected_theme]["bgcolor"],
                row=1,
                col=2,
            )

            # 更新曲面颜色主题
            fig.data[2].colorscale = scene_settings[selected_theme].get(
                "colorscale", "Viridis"
            )

            # 调整整体布局
            fig.update_layout(
                # height=800,
                showlegend=False,
                margin=dict(l=50, r=50, b=50, t=50),
            )

            col2.plotly_chart(fig, use_container_width=True)

# 工艺及设备参数 (tab2)
with tab2:
    st.header("连铸工艺参数设置")

    # 创建两列布局
    col1, col2 = st.columns([0.4, 0.6], gap="medium")

    with col1:
        # 结晶器参数容器
        with st.container(border=True):
            st.subheader("结晶器参数")
            casting_temperature = st.number_input(
                "连铸钢水温度", min_value=0.0, value=1550.0, step=10.0
            )
            rowup = st.columns(3, vertical_alignment="center")
            with rowup[0]:
                st.latex(
                    "q = A-B \\sqrt{t}",
                    help="结晶器区的边界条件为热流密度边界条件，t为钢在结晶器中的时间。W/m²",
                )
            with rowup[1]:
                q_mold_a = (
                    st.number_input(
                        "A参数数值",
                        min_value=0.0,
                        value=2680.0,
                        step=10.0,
                        help="A参数",
                    )
                    * 1000
                )
            with rowup[2]:
                q_mold_b = (
                    st.number_input(
                        "B参数数值",
                        min_value=0.0,
                        value=276.0,
                        step=10.0,
                        help="B参数",
                    )
                    * 1000
                )
            # 第一行3列
            row1 = st.columns(3)
            with row1[0]:
                casting_speed = st.number_input(
                    "拉坯速度 (m/min)", min_value=0.1, value=1.2, step=0.1
                )
            with row1[1]:
                heat_flow_factor = st.number_input(
                    "热流密度修正系数", min_value=0.1, value=1.0, step=0.1
                )
            with row1[2]:
                width = (
                    st.number_input(
                        "断面宽度 (mm)", min_value=100.0, value=1000.0, step=10.0
                    )
                    / 1000
                )

            # 第二行3列
            row2 = st.columns(3)
            with row2[0]:
                thickness = (
                    st.number_input(
                        "断面厚度 (mm)", min_value=50.0, value=200.0, step=5.0
                    )
                    / 1000
                )
            with row2[1]:
                steel_height = (
                    st.number_input(
                        "钢液高度 (mm)", min_value=100.0, value=500.0, step=10.0
                    )
                    / 1000
                )
            with row2[2]:
                mold_length = (
                    st.number_input(
                        "结晶器高度 (mm)", min_value=700.0, value=1000.0, step=100.0
                    )
                    / 1000
                )

        # 二冷区参数容器
        with st.container(border=True):
            st.subheader("二冷区参数")

            # 从results文件夹中的cooling_formulas.json文件夹中读取enabled=True的公式的name的值以供选择。反馈到cooling_formula_name变量。
            cooling_formulas_path = os.path.join(
                os.path.dirname(__file__), "resources", "cooling_formulas.json"
            )
            with open(cooling_formulas_path, "r", encoding="utf-8") as f:
                cooling_data = json.load(f)
                cooling_formula_name = st.selectbox(
                    "二冷区换热系数计算公式", cooling_data["enabled_formulas"]
                )
            if "spray_zones" not in st.session_state:
                st.session_state.spray_zones = 5
                st.session_state.width_flows = [20.0] * st.session_state.spray_zones
                st.session_state.thickness_flows = [20.0] * st.session_state.spray_zones
                st.session_state.water_temps = [25.0] * st.session_state.spray_zones
                st.session_state.segment_length = [
                    steel_height
                ] * st.session_state.spray_zones

            def update_spray_zones():
                new_count = st.session_state.spray_zones_input
                current_count = st.session_state.spray_zones

                if new_count > current_count:
                    # 增加分区
                    for _ in range(new_count - current_count):
                        st.session_state.width_flows.append(20.0)
                        st.session_state.thickness_flows.append(20.0)
                        st.session_state.water_temps.append(25.0)
                        st.session_state.segment_length.append(steel_height)
                elif new_count < current_count:
                    # 减少分区
                    st.session_state.width_flows = st.session_state.width_flows[
                        :new_count
                    ]
                    st.session_state.thickness_flows = st.session_state.thickness_flows[
                        :new_count
                    ]
                    st.session_state.water_temps = st.session_state.water_temps[
                        :new_count
                    ]
                    st.session_state.segment_length = st.session_state.segment_length[
                        :new_count
                    ]

                st.session_state.spray_zones = new_count

            zone_col1, zone_col2 = st.columns([0.5, 0.5], vertical_alignment="bottom")
            with zone_col1:
                spray_zones = st.number_input(
                    "分区数目",
                    min_value=1,
                    value=st.session_state.spray_zones,
                    step=1,
                    key="spray_zones_input",
                    on_change=update_spray_zones,
                )
            with zone_col2:
                tab2_submit_button = st.button(
                    "保存并更新分区信息",
                    use_container_width=True,
                    type="primary",
                )

            # 为每个分区创建容器
            for i in range(spray_zones):
                with st.container(border=True):
                    cols = st.columns(
                        [5, 5, 5, 5], gap="small", vertical_alignment="bottom"
                    )

                    # 水量输入
                    with cols[0]:

                        def update_width_flow(i):
                            st.session_state.width_flows[i] = st.session_state[
                                f"width_flow_{i}"
                            ]

                        st.number_input(
                            f"{i}区顶面水量 (L/min)",
                            min_value=0.0,
                            max_value=100.0,
                            value=st.session_state.width_flows[i],
                            step=0.5,
                            key=f"width_flow_{i}",
                            on_change=update_width_flow,
                            args=(i,),
                        )

                    # 厚度面水量
                    with cols[1]:

                        def update_thickness_flow(i):
                            st.session_state.thickness_flows[i] = st.session_state[
                                f"thickness_flow_{i}"
                            ]

                        st.number_input(
                            f"{i}区侧面水量 (L/min)",
                            min_value=0.0,
                            max_value=100.0,
                            value=st.session_state.thickness_flows[i],
                            step=0.5,
                            key=f"thickness_flow_{i}",
                            on_change=update_thickness_flow,
                            args=(i,),
                        )

                    # 水温输入
                    with cols[2]:

                        def update_water_temp(i):
                            st.session_state.water_temps[i] = st.session_state[
                                f"water_temp_{i}"
                            ]

                        st.number_input(
                            f"{i}区水温 (°C)",
                            min_value=10.0,
                            value=st.session_state.water_temps[i],
                            step=1.0,
                            key=f"water_temp_{i}",
                            on_change=update_water_temp,
                            args=(i,),
                        )

                    # 位置输入
                    with cols[3]:

                        def update_water_location(i):
                            st.session_state.segment_length[i] = st.session_state[
                                f"water_location_{i}"
                            ]

                        st.number_input(
                            f"{i}区长度 (mm)",
                            min_value=0.0,
                            value=st.session_state.segment_length[i],
                            step=10.0,
                            key=f"water_location_{i}",
                            on_change=update_water_location,
                            args=(i,),
                        )
        # 空冷区参数容器
        with st.container(border=True):
            st.subheader("空冷区参数")
            air_cols = st.columns([1, 1, 1])
            with air_cols[0]:
                emissivity = st.number_input(
                    "辐射率",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.01,
                    help="钢坯表面辐射率，范围0-1",
                )
            with air_cols[1]:
                ambient_temp = st.number_input(
                    "环境温度(℃)", value=0.0, step=5.0, help="空冷区环境温度"
                )
            with air_cols[2]:
                air_cooling_length = st.number_input(
                    "空冷区长度(m)",
                    min_value=0.0,
                    value=3.0,
                    step=0.1,
                    help="空冷区总长度",
                )

    with col2:
        tab2_show = st.container(border=True)
        # 参数显示容器
        if tab2_submit_button:
            # 收集所有工艺参数
            process_data = {
                "mold_parameters": {
                    "casting_temperature": casting_temperature,
                    "casting_speed": casting_speed,
                    "heat_flow_factor": heat_flow_factor,
                    "slab_width": width,
                    "slab_thickness": thickness,
                    "liquid_steel_height": steel_height,
                    "mold_length": mold_length,
                    "heat_flux_a": q_mold_a,
                    "heat_flux_b": q_mold_b,
                },
                "spray_parameters": {
                    "zone_count": spray_zones,
                    "width_flows": st.session_state.width_flows,
                    "thickness_flows": st.session_state.thickness_flows,
                    "water_temps": st.session_state.water_temps,
                    "segment_length": st.session_state.segment_length,
                    "cooling_formula": cooling_formula_name,
                },
                "air_cooling_parameters": {
                    "emissivity": emissivity,
                    "ambient_temp": ambient_temp,
                    "cooling_length": air_cooling_length,
                },
            }

            with tab2_show:
                st.subheader("结晶器参数汇总")
                tab2_show_cols = st.columns([2, 3], gap="medium", border=True)
                with tab2_show_cols[0]:
                    st.dataframe(
                        [
                            {
                                "参数": "连铸钢水温度",
                                "值": casting_temperature,
                                "单位": "℃",
                            },
                            {"参数": "钢坯宽度", "值": width, "单位": "mm"},
                            {"参数": "钢坯厚度", "值": thickness, "单位": "mm"},
                            {"参数": "钢液高度", "值": steel_height, "单位": "mm"},
                            {"参数": "结晶器高度", "值": mold_length, "单位": "mm"},
                            {"参数": "拉坯速度", "值": casting_speed, "单位": "m/min"},
                        ]
                    )
                with tab2_show_cols[1]:
                    mold_total_time = mold_length / 1000 / casting_speed * 60
                    plotter = HeatTransferCalculator()
                    fig = plotter.plot_mold_heat_flux(
                        t_max=mold_total_time, a=q_mold_a, b=q_mold_b
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # 二冷区参数显示
                st.subheader("二冷区参数汇总")
                tab2_coolings_cols = st.columns([3, 2], gap="medium", border=True)
                with tab2_coolings_cols[0]:
                    # 计算各二冷区换热系数
                    htc_calculator = HeatTransferCalculator()
                    h_values = []
                    for i in range(spray_zones):
                        # 计算顶面和侧面换热系数
                        # 计算区域长度(转换为m)
                        zone_length = st.session_state.segment_length[i]  # mm to m

                        # 计算顶面比水量 (L/(m²·s))
                        top_area = (zone_length) * (width)  # m²
                        top_water_flow = (
                            st.session_state.width_flows[i] / 60
                        )  # L/min to L/s
                        V_top = top_water_flow / top_area

                        # 计算侧面比水量 (L/(m²·s))
                        side_area = (zone_length) * (thickness)  # m²
                        side_water_flow = (
                            st.session_state.thickness_flows[i] / 60
                        )  # L/min to L/s
                        V_side = side_water_flow / side_area

                        # 计算换热系数
                        h_top = round(
                            htc_calculator.secondary_cooling_h(
                                V=V_top,
                                T_w=st.session_state.water_temps[i],
                                method=cooling_formula_name,
                            )
                            * 1000,
                            4,
                        )
                        h_side = round(
                            htc_calculator.secondary_cooling_h(
                                V=V_side,
                                T_w=st.session_state.water_temps[i],
                                method=cooling_formula_name,
                            )
                            * 1000,
                            4,
                        )
                        h_values.append({"top": h_top, "side": h_side})

                    # 更新process_data
                    process_data["spray_parameters"][
                        "heat_transfer_coefficients"
                    ] = h_values

                    # 确保results目录存在
                    os.makedirs("results", exist_ok=True)

                    # 保存为JSON文件
                    with open("results/process_data.json", "w", encoding="utf-8") as f:
                        json.dump(process_data, f, ensure_ascii=False, indent=4)

                    # 绘制换热系数分布直方图
                    fig = htc_calculator.plot_heat_transfer_coefficients(
                        mold_exit_time=0,
                        last_zone_end_time=st.session_state.segment_length,
                        h_values=h_values,
                        zone_names=[f"{i+1}区" for i in range(spray_zones)],
                        casting_speed=casting_speed,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tab2_coolings_cols[1]:
                    # 显示当前选择的冷却公式
                    cooling_formulas_path = os.path.join(
                        os.path.dirname(__file__), "resources", "cooling_formulas.json"
                    )
                    with open(cooling_formulas_path, "r", encoding="utf-8") as f:
                        cooling_data = json.load(f)
                        for formula in cooling_data["formulas"]:
                            if formula["name"] == cooling_formula_name:
                                st.caption(f"{formula['description']}表达式如下：")
                                # 使用sympy优化公式显示
                                # 目前这里还有一定问题，后续再进行调整
                                try:
                                    # 直接使用原始公式字符串
                                    formula_str = formula["formula_latex"]
                                    st.latex(formula_str)
                                except Exception as e:
                                    st.warning(f"公式解析错误: {e}")
                                    display_str = formula["formula"]
                                    if "unit" in formula:
                                        display_str += f" (单位: {formula['unit']})"
                                    st.latex(display_str)
                                if "parameters" in formula:
                                    params = "\n".join(
                                        [
                                            f"{k}: {v}"
                                            for k, v in formula["parameters"].items()
                                        ]
                                    )
                                    st.caption(f"参数:\n{params}")

                    spray_data = process_data["spray_parameters"]
                    water_data = {
                        "区域编号": [f"{i}区" for i in range(spray_data["zone_count"])],
                        "宽度面水量(L/min)": spray_data["width_flows"],
                        "厚度面水量(L/min)": spray_data["thickness_flows"],
                        "水温(℃)": spray_data["water_temps"],
                        "长度(mm)": spray_data["segment_length"],
                    }
                    st.dataframe(
                        water_data,
                        column_config={
                            "区域编号": st.column_config.TextColumn("区域"),
                            "宽度面水量(L/min)": st.column_config.NumberColumn(
                                "宽度面水量", format="%.1f L/min"
                            ),
                            "厚度面水量(L/min)": st.column_config.NumberColumn(
                                "厚度面水量", format="%.1f L/min"
                            ),
                            "水温(℃)": st.column_config.NumberColumn(
                                "水温", format="%.1f ℃"
                            ),
                            "位置(mm)": st.column_config.NumberColumn(
                                "位置", format="%.0f mm"
                            ),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.divider()
                    st.caption("说明：")
                    st.caption(
                        "1. 如果换热系数分布图标没有显示，则说明该公式并不适用于这一冷却环境，可能的原因主要是冷却水量过大或过小导致的，更换其他公式即可。"
                    )
                # 空冷区参数显示
                st.subheader("空冷区参数汇总")
                air_cooling_cols = st.columns([2, 3], border=True)
                with air_cooling_cols[0]:
                    # 显示空冷区参数表格
                    st.latex("q=\\epsilon \\sigma (T_{s}^{4}-T_{a}^{4})")
                    st.caption("$\\epsilon $ 为辐射率")
                    st.caption("$\\sigma$ 为Stefan-Boltzmann常数")
                    st.caption("$T_{s}$ 为钢表面温度")
                    st.caption("$T_{a}$ 为环境温度")
                    air_data = process_data["air_cooling_parameters"]
                    st.dataframe(
                        {
                            "参数": ["辐射率", "环境温度", "空冷区长度"],
                            "值": [
                                air_data["emissivity"],
                                air_data["ambient_temp"],
                                air_data["cooling_length"],
                            ],
                            "单位": ["-", "℃", "m"],
                        },
                        column_config={
                            "参数": st.column_config.TextColumn("参数名称"),
                            "值": st.column_config.NumberColumn(
                                "参数值", format="%.2f"
                            ),
                            "单位": st.column_config.TextColumn("单位"),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )
                with air_cooling_cols[1]:
                    # 绘制空冷区热流密度随温度变化曲线
                    air_data = process_data["air_cooling_parameters"]
                    temps = np.linspace(
                        1200, air_data["ambient_temp"], 100
                    )  # 温度从1200℃到环境温度
                    heat_fluxes = [
                        HeatTransferCalculator().air_cooling_heat_flux(
                            T_s=T,
                            T_a=air_data["ambient_temp"],
                            emissivity=air_data["emissivity"],
                        )
                        for T in temps
                    ]

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=temps,
                            y=heat_fluxes,
                            mode="lines",
                            name="空冷区热流密度",
                            line=dict(color="royalblue", width=2),
                            hovertemplate="温度: %{x:.1f}℃<br>热流密度: %{y:.2f} kW/m²<extra></extra>",
                        )
                    )

                    fig.update_layout(
                        title="空冷区热流密度随温度变化",
                        xaxis_title="温度 (℃)",
                        yaxis_title="热流密度 (kW/m²)",
                        xaxis=dict(showgrid=True, gridcolor="lightgray"),
                        yaxis=dict(showgrid=True, gridcolor="lightgray"),
                        plot_bgcolor="white",
                        height=400,
                    )

                    fig.update_xaxes(autorange="reversed")  # 反转x轴显示
                    st.plotly_chart(fig, use_container_width=True)

    # Lx,
    # Ly,
    # nx,
    # ny,
    # h_top,
    # h_right,
    # T_inf_top,
    # T_inf_right,
    # dt,
    # total_time,
    # initial_temp=1550,  # 使用全局定义的初始温度
    # tol=1e-6,  # 使用全局定义的容差
# 计算参数 (tab3)
with tab3:
    st.header("计算控制参数")
    tab3_col1, tab3_col2, tab3_col3 = st.columns([3, 14, 3], gap="medium", border=True)

    with tab3_col1:
        st.subheader("计算设置")
        mesh_size = (
            st.number_input(
                "空间步长 (mm)",
                min_value=1.0,
                value=20.0,
                step=5.0,
                help="建议值20-50mm，值越小计算量越大",
            )
            / 1000
        )
        time_step = st.number_input("时间步长 (s)", min_value=0.1, value=1.0, step=0.1)
        if st.button("模型建立及初始化", key="init", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("正在加载工艺数据...")
                init_data = initialize_model(grid_size=mesh_size, time_step=time_step)
                progress_bar.progress(50)

                status_text.text("正在计算参数...")
                progress_bar.progress(80)

                status_text.text("正在保存结果...")
                progress_bar.progress(100)
                status_text.text("初始化完成!")

            except Exception as e:
                progress_bar.empty()
                st.error(f"模型初始化失败: {str(e)}")
        if st.button("开始计算", key="calc", use_container_width=True, type="primary"):
            try:
                from calculation_processor import process_segments

                with st.spinner("计算中，请稍候..."):
                    results = process_segments()
                st.success(f"计算完成！共处理了{len(results)}段数据")
                for i, result in enumerate(results):
                    st.write(f"第{i+1}段结果保存至: {result['output_path']}")
            except Exception as e:
                st.error(f"计算出错: {str(e)}")
    # with tab3_col2:
    #     st.subheader("计算结果")
    #     # 预留热图显示区域
    #     st.write("结晶器进出口热图:")
    #     # st.image("placeholder.png")  # 替换为实际图像路径

    #     # st.image("placeholder.png")  # 替换为实际图像路径
    # with tab3_col3:
    #     st.subheader("结果查看")
