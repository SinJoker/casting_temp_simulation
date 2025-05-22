# 必须在最前面调用set_page_config
import streamlit as st
import pandas as pd
import json
import os
from thermal_properties import (
    calculate_const_properties,
    calculate_liquidus_temp,
    calculate_solidus_temp,
)
from prop_vs_temp import (
    cp_cal,
    lamda_cal,
    rho_cal,
)

st.set_page_config(layout="wide")

# """设置连铸区温度场模拟的UI界面"""
st.title("连铸区温度场模拟计算")

# 初始化session state
if "components" not in st.session_state:
    st.session_state.components = []

tab1, tab2, tab3 = st.tabs(["钢物性参数", "工艺及设备参数", "计算参数"])

with tab1:
    # 从json文件读取钢种物性参数和预设元素
    json_path = os.path.join(os.path.dirname(__file__), "steel_properties.json")
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
        # 从文件formula_names.json读取液相线和固相线计算公式列表
        with open("formula_names.json", "r", encoding="utf-8") as f:
            formula_data = json.load(f)
            liquidus_formulas = formula_data["liquidus_formulas"]
            solidus_formulas = formula_data["solidus_formulas"]

        liquid_formula = st.selectbox("液相线的计算公式", liquidus_formulas)
        solid_formula = st.selectbox("固相线的计算公式", solidus_formulas)

    with col2:

        if (
            tab1_col2.button("保存并更新成分", use_container_width=True)
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

            # 绘制物性参数图表(使用plotly)
            import numpy as np
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # 获取物性参数
            props = const_results["const_properties"]
            Tl = const_results["liquid_temp"]
            Ts = const_results["solid_temp"]
            Tc = Tl + 100  # 假设临界温度比液相线高100℃

            # 创建温度范围(1600~1000℃)
            temps = np.linspace(1600, 1000, 100)
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

            # 创建上下布局的图表
            fig = make_subplots(
                rows=2,
                cols=1,
                specs=[[{"type": "xy"}], [{"type": "surface"}]],
                subplot_titles=("比热容和密度随温度变化", "导热系数随温度和距离变化"),
                vertical_spacing=0.15,
            )

            # 上方图表: 比热容和密度
            fig.add_trace(
                go.Scatter(
                    x=temps, y=cps, name="比热容 (J/kg·K)", line=dict(color="red")
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=temps,
                    y=rhos,
                    name="密度 (kg/m³)",
                    line=dict(color="blue"),
                    yaxis="y2",
                ),
                row=1,
                col=1,
            )

            fig.update_layout(
                xaxis=dict(title="温度 (℃)", range=[1600, 1000]),
                yaxis=dict(title="比热容 (J/kg·K)", titlefont=dict(color="red")),
                yaxis2=dict(
                    title="密度 (kg/m³)",
                    titlefont=dict(color="blue"),
                    overlaying="y",
                    side="right",
                ),
            )

            # 下方图表: 导热系数3D图
            fig.add_trace(
                go.Surface(
                    x=T_grid, y=P_grid, z=lamdas, colorscale="Viridis", name="导热系数"
                ),
                row=2,
                col=1,
            )

            fig.update_layout(
                scene=dict(
                    xaxis_title="温度 (℃)",
                    yaxis_title="距离 (m)",
                    zaxis_title="导热系数 (W/m·K)",
                ),
                height=800,
            )

            tab1_col2.plotly_chart(fig, use_container_width=True)

# # 工艺及设备参数 (tab2)
# with tab2:
#     st.header("连铸工艺参数设置")
#     col1, col2, col3 = st.columns([0.15, 0.15, 0.7])

#     with col1:
#         st.subheader("结晶器参数")
#         casting_speed = st.number_input(
#             "拉坯速度 (m/min)", min_value=0.1, value=1.2, step=0.1
#         )
#         heat_flow_factor = st.number_input(
#             "热流密度修正系数", min_value=0.1, value=1.0, step=0.1
#         )
#         width = st.number_input("断面宽度 (mm)", min_value=100, value=1000, step=10)
#         thickness = st.number_input("断面厚度 (mm)", min_value=50, value=200, step=5)
#         steel_height = st.number_input(
#             "钢液高度 (mm)", min_value=100, value=500, step=10
#         )
#         mold_length = st.number_input(
#             "结晶器高度 (mm)", min_value=700, value=1000, step=100
#         )

#     with col2:
#         st.subheader("二冷区参数")
#         spray_zones = st.number_input("分区数目", min_value=1, value=5, step=1)
#         water_flows = [
#             st.number_input(
#                 f"分区{i+1}水量 (L/min)", min_value=0.0, value=20.0, step=1.0
#             )
#             for i in range(spray_zones)
#         ]
#         water_temps = [
#             st.number_input(f"分区{i+1}水温 (°C)", min_value=10.0, value=25.0, step=1.0)
#             for i in range(spray_zones)
#         ]

#     with col3:
#         st.subheader("参数显示")
#         st.write(f"当前拉坯速度: {casting_speed} m/min")
#         st.write(f"结晶器长度: {mold_length} mm")
#         st.write("二冷区水量分配:")
#         for i in range(spray_zones):
#             st.write(f"分区{i+1}: {water_flows[i]} L/min, {water_temps[i]} °C")

#         # 预留图形显示区域
#         st.subheader("二冷区结构示意图")
#         # st.image("placeholder.png")  # 替换为实际图像路径

#         if st.button("保存工艺参数"):
#             st.rerun()

# # 计算参数 (tab3)
# with tab3:
#     st.header("计算控制参数")
#     col1, col2 = st.columns([0.15, 0.85])

#     with col1:
#         st.subheader("计算设置")
#         mesh_size = st.number_input("空间步长 (mm)", min_value=1, value=10, step=1)
#         time_step = st.number_input("时间步长 (s)", min_value=0.1, value=1.0, step=0.1)
#         init_btn = st.button("初始化")
#         calc_btn = st.button("开始计算")

#     with col2:
#         st.subheader("计算结果")
#         # 预留热图显示区域
#         st.write("结晶器进出口热图:")
#         # st.image("placeholder.png")  # 替换为实际图像路径

#         if st.button("开始计算"):
#             st.rerun()
#         st.write("各二冷区分区热图:")
#         # st.image("placeholder.png")  # 替换为实际图像路径

# # return1 = {
# #     "steel_properties": steel_properties,
# #     "tab1": tab1,
# #     "tab2": tab2,
# #     "tab3": tab3,
# #     "kind": kind,
# #     "t_l": t_l,
# #     "t_s": t_s,
# #     "casting_speed": casting_speed,
# #     "mold_length": mold_length,
# #     "spray_zones": spray_zones,
# #     "water_flow": water_flow,
# #     "spray_density": spray_density,
# #     "water_temp": water_temp,
# #     "mesh_size": mesh_size,
# #     "time_step": time_step,
# #     "max_iter": max_iter,
# #     "tolerance": tolerance,
# #     "auto_save": auto_save,
# # }
