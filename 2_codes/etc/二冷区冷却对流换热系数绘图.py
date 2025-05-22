import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_bidirectional_histogram(
    x_data,
    y1_data,
    y2_data,
    titles=("Upper Distribution", "Lower Distribution"),
    colors=("blue", "red"),
    bin_size=5,
    height=600,
):
    """
    绘制双向直方图（上下子图）

    参数：
    x_data : list/array - 用于分箱的X轴数据（不直接用于绘图）
    y1_data : list/array - 上子图数据
    y2_data : list/array - 下子图数据
    titles : tuple - 子图标题 (上标题, 下标题)
    colors : tuple - 颜色配置 (上颜色, 下颜色)
    bin_size : int - 直方图分箱宽度
    height : int - 画布总高度
    """
    # 自动计算分箱范围
    combined = np.concatenate([y1_data, y2_data])
    x_min, x_max = np.floor(combined.min()), np.ceil(combined.max())

    # 创建子图框架
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, subplot_titles=titles
    )

    # 添加上方直方图
    fig.add_trace(
        go.Histogram(
            x=y1_data,
            name=titles[0],
            marker_color=colors[0],
            xbins=dict(start=x_min, end=x_max, size=bin_size),
        ),
        row=1,
        col=1,
    )

    # 添加下方直方图
    fig.add_trace(
        go.Histogram(
            x=y2_data,
            name=titles[1],
            marker_color=colors[1],
            xbins=dict(start=x_min, end=x_max, size=bin_size),
        ),
        row=2,
        col=1,
    )

    # 统一坐标轴配置
    fig.update_layout(
        height=height,
        showlegend=False,
        bargap=0.05,
        xaxis=dict(title="Value Range"),
        yaxis=dict(title="Count"),
        yaxis2=dict(title="Count"),
    )

    # 添加分箱参考线
    for bin_edge in np.arange(x_min, x_max + bin_size, bin_size):
        fig.add_vline(
            x=bin_edge, line_width=1, line_dash="dot", line_color="gray", opacity=0.5
        )

    return fig


# 生成示例数据
np.random.seed(42)
data_x = np.linspace(0, 100, 10)  # 用于分箱的参考数据
y1 = np.random.normal(50, 15, 20)
y2 = np.random.normal(70, 10, 20)

# 绘制图形
fig = plot_bidirectional_histogram(
    x_data=data_x,
    y1_data=y1,
    y2_data=y2,
    titles=("顶面对流换热系数", "侧面对流换热系数"),
    colors=("#D77071", "#6888F5"),
    bin_size=5,
)
fig.show()
