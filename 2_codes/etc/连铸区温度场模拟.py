import numpy as np
import streamlit as st
import plotly.graph_objects as go
from ui import setup_ui


def main():
    # 获取UI组件和参数
    ui_data = setup_ui()
    steel_properties = ui_data["steel_properties"]
    kind = ui_data["kind"]
    t_l = ui_data["t_l"]
    t_s = ui_data["t_s"]
    tab2 = ui_data["tab2"]
    tab3 = ui_data["tab3"]

    # 从UI获取计算参数
    pouring_temp = ui_data.get("pouring_temp", 1550)
    casting_speed = ui_data.get("casting_speed", 1.2)
    width = ui_data.get("width", 200)
    thickness = ui_data.get("thickness", 20)
    space_step = ui_data.get("space_step", 5.0)
    time_step = ui_data.get("time_step", 0.1)

    with tab2:
        # 显示工艺参数
        st.write(f"钢种: {kind}")
        st.write(f"液相线温度: {t_l:.1f}℃")
        st.write(f"固相线温度: {t_s:.1f}℃")
        st.write(f"拉坯速度: {casting_speed} m/min")

    with tab3:
        # 初始化计算
        if st.button("初始化计算"):
            nx = int(82.5 / space_step) + 1
            ny = int(82.5 / space_step) + 1
            st.session_state.T = np.ones((ny, nx)) * pouring_temp
            st.session_state.initialized = True
            st.success(f"初始化完成，网格尺寸: {nx}x{ny}")

        # 温度场计算
        if st.button("开始计算") and "initialized" in st.session_state:
            progress_bar = st.progress(0)
            T = st.session_state.T
            nx, ny = T.shape

            # 简化的热传导计算
            for _ in range(100):
                new_T = T.copy()
                for i in range(1, ny - 1):
                    for j in range(1, nx - 1):
                        new_T[i, j] = 0.25 * (
                            T[i + 1, j] + T[i - 1, j] + T[i, j + 1] + T[i, j - 1]
                        )
                T = new_T
                progress_bar.progress(_ / 100)

            st.session_state.T = T
            st.session_state.calculated = True
            st.success("计算完成")

        # 显示结果
        if "calculated" in st.session_state:
            fig = go.Figure(
                data=go.Contour(
                    z=st.session_state.T,
                    colorscale="Hot",
                    colorbar=dict(title="温度(℃)"),
                )
            )
            fig.update_layout(
                title="连铸坯1/4截面温度场分布",
                xaxis_title="宽度方向(mm)",
                yaxis_title="厚度方向(mm)",
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
