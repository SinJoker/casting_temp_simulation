import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def compute_temperatures(composition):
    """
    输入成分字典（例如：{'C': 0.1, 'Si': 0.2, ...}），返回不同公式的计算结果。
    支持的成分元素：C, Si, Mn, P, S, Ni, Cr, Mo, Cu, V, W, N, Al, Co, Fe等。
    """
    results = {"Liquidus": {}, "Solidus": {}}

    # 提取成分，缺失元素默认设为0
    def get_value(key):
        return composition.get(key, 0.0)

    # 完整元素提取(所有可能用到的元素)
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
    Fe = get_value("Fe")
    Ti = get_value("Ti")
    O = get_value("O")
    Nb = get_value("Nb")
    Ta = get_value("Ta")
    As = get_value("As")
    Sn = get_value("Sn")
    Zr = get_value("Zr")

    # -------------------------- 液相线公式 --------------------------
    # 1. 1980年《国外连铸新技术》译文集(二)第14页
    results["Liquidus"]["1980_国外连铸新技术"] = 1535 - (
        78 * C + 4.9 * Mn + 3.6 * Al + 7.6 * Si
    )

    # 2. 1988年商家K提供给某钢铁公司超低头板坯连铸机
    def liquidus_1988():
        base = 1536.6 - (
            8 * Si + 5 * Mn + 5.5 * Cu + 1.5 * Cr + 4 * Ni + 2 * Mo + 18 * Ti + 1
        )
        if C <= 0.5:
            return base - 88 * C
        else:
            return base - (0.5 * 88 + (C - 0.5) * 76)

    results["Liquidus"]["1988_商家K"] = liquidus_1988()

    # 3. 1989年《日本广田连铸技术》武钢二炼钢刘良春等编译第91页
    results["Liquidus"]["1989_日本广田连铸技术"] = 1536.6 - (
        88 * C
        + 25 * S
        + 5 * Cr
        + 8 * Si
        + 5 * Mn
        + 30 * P
        + 2 * Mo
        + 4 * Ni
        + 18 * Ti
        + 2 * V
    )

    # 4. 1990年《连续铸钢手册》冶金工业出版社81页 - 通用公式
    results["Liquidus"]["1990_连续铸钢手册_通用"] = 1539 - (
        70 * C + 8 * Si + 5 * Mn + 30 * P + 25 * S + Cu + 4 * Ni + 1.5 * Cr
    )

    # 5. 1990年《连续铸钢手册》冶金工业出版社81页 - 碳素钢公式
    def liquidus_1990_carbon():
        if C < 0.6:
            return 1539 - (
                90 * C
                + 6.2 * Si
                + 1.7 * Mn
                + 28 * P
                + 40 * S
                + 2.6 * Cu
                + 2.9 * Ni
                + 1.8 * Cr
                + 5.1 * Al
            )
        else:
            return 1534 - (
                73 * C
                + 12 * Si
                + 3 * Mn
                + 28 * P
                + 30 * S
                + 7 * Cu
                + 3.5 * Ni
                + Cr
                + 3 * Al
            )

    results["Liquidus"]["1990_连续铸钢手册_碳素钢"] = liquidus_1990_carbon()

    # 6. 1990年《连续铸钢手册》冶金工业出版社81页 - 特殊钢种
    results["Liquidus"]["1990_连续铸钢手册_特殊钢"] = 1534 - (
        91 * C + 21 * Si + 3.5 * Mn + 4 * Ni + 0.65 * Cr + 3 * Mo
    )

    # 7. 1994年《连续铸钢原理与工艺》
    results["Liquidus"]["1994_连续铸钢原理与工艺"] = 1537 - (
        88 * C
        + 8 * Si
        + 5 * Mn
        + 30 * P
        + 25 * S
        + 5 * Cu
        + 4 * Ni
        + 2 * Mo
        + 2 * V
        + 1.5 * Cr
    )

    # 8. 1997年《宝钢技术》第3期第20页日本人的"平居公式"
    results["Liquidus"]["1997_宝钢技术_平居公式"] = 1538 - (
        55 * C + 80 * C**2 + 13 * Si + 4.8 * Mn + 1.5 * Cr + 4.3 * Ni + 30 * P + 30 * S
    )

    # 9. 2000年商家D给T钢铁公司不锈钢连铸机推荐公式
    results["Liquidus"]["2000_商家D_不锈钢1"] = 1536.6 - (
        90 * C
        + 8 * Si
        + 5 * Mn
        + 30 * P
        + 25 * S
        + 3 * Al
        + 1.55 * Cr
        + 4 * Ni
        + 2 * Mo
        + 18 * Ti
        + 80 * N
        + 5 * Cu
    )
    results["Liquidus"]["2000_商家D_不锈钢2"] = 1536.6 - (
        90 * C
        + 8 * Si
        + 5 * Mn
        + 30 * P
        + 25 * S
        + 5 * Cu
        + 4 * Ni
        + 1.55 * Cr
        + 2 * Mo
        + 18 * Ti
        + 80 * 0.0001 * (N * 10000 - 100)  # ppm转换
    )

    # 10. 2006年《薄板坯连铸连轧钢的组织性能控制》
    results["Liquidus"]["2006_薄板坯连铸连轧"] = 1537 - (
        65 * C + 8 * Si + 5 * Mn + 30 * P + 25 * S + 2.7 * Al + 80 * O + 90 * N
    )

    # 11. 2013年《炼钢》第5期第66~67页通用公式
    results["Liquidus"]["2013_炼钢_通用"] = 1539 - (
        70 * C + 8 * Si + 5 * Mn + 30 * P + 25 * S + 4 * Ni + 1.5 * Cr
    )

    # 12. 1998年商家A提供给A钢铁公司板坯连铸机镀锡板计算公式
    results["Liquidus"]["1998_商家A_镀锡板"] = 1536.8 - (
        88 * C
        + 5 * Mn
        + 8 * Si
        + 30 * P
        + 25 * S
        + 5 * Cr
        + 2 * Mo
        + 4 * Ni
        + 18 * Ti
        + 2 * V
    )

    # 13. 2000年商家B给A钢铁公司不锈钢连铸机推荐公式
    def liquidus_2000_stainless():
        if C < 0.2:
            return 1536 - (
                65 * C
                + 8 * Si
                + 5 * Mn
                + 30 * P
                + 25 * S
                + 1.7 * Al
                + 5 * Cu
                + 1.5 * Cr
                + 4 * Ni
                + 2 * V
                + W
                + 1.7 * Co
                + 12.8 * Zr
                + 7 * Nb
                + 3 * Ta
                + 14 * Ti
                + 14 * As
                + 10 * Sn
            )
        elif 0.2 <= C <= 0.5:
            return 1536 - (
                88 * C
                + 8 * Si
                + 5 * Mn
                + 30 * P
                + 25 * S
                + 1.7 * Al
                + 5 * Cu
                + 1.5 * Cr
                + 4 * Ni
                + 2 * V
            )
        else:
            return 1536 - (
                9 * C
                + 65 * C**2
                + 10 * Si
                + 6 * Mn
                + 30 * P
                + 30 * S
                + 3 * Al
                + 5 * Cu
                + 1.5 * Cr
                + 3.5 * Ni
                + 2 * V
            )

    results["Liquidus"]["2000_商家B_不锈钢"] = liquidus_2000_stainless()

    # 14. 2003年《武钢炼钢生产技术进步概况》硅钢公式
    results["Liquidus"]["2003_武钢_硅钢"] = 1539 - (
        65 * C
        + 8 * Si
        + 5 * Mn
        + 30 * P
        + 25 * S
        + 5 * Cu
        + 4 * Ni
        + 2 * Mo
        + 2 * V
        + 1.5 * Cr
    )

    # 15. 2007年《品种钢、优特钢连铸900问》
    results["Liquidus"]["2007_连铸900问"] = 1539 - (
        70 * C + 8 * Si + 5 * Mn + 30 * P + 25 * S + 4 * Ni + 2 * Mo + 2 * V
    )

    # 16. 商家D碳钢板坯连铸机公式(分段碳系数)
    def liquidus_vendorD_carbon():
        if C <= 0.025:
            factor = 90.0
        elif 0.025 < C <= 0.050:
            factor = 82.0
        elif 0.050 < C <= 0.101:
            factor = 86.0
        elif 0.101 < C <= 0.500:
            factor = 88.4
        elif 0.500 < C <= 0.600:
            factor = 86.1
        elif 0.600 < C <= 0.700:
            factor = 84.2
        elif 0.700 < C <= 0.800:
            factor = 83.2
        else:
            factor = 82.3
        return 1536.6 - (
            factor * C
            + 8 * Si
            + 5 * Mn
            + 30 * P
            + 25 * S
            + 2 * Ti
            + 2 * Mo
            + 5 * Cu
            + 4 * Ni
            + 1.5 * Cr
            + 5.1 * Al
            + 90 * N
        )

    results["Liquidus"]["商家D_碳钢分段"] = liquidus_vendorD_carbon()

    # 17. 铸铁液相线温度
    results["Liquidus"]["铸铁"] = 1650 - 124.5 * C - 26.7 * (Si + 2.45 * P)

    # -------------------------- 固相线公式 --------------------------
    # 1. 1997年《宝钢技术》第3期第20页日本人的"平居公式"
    def solidus_1997():
        base = (
            20.5 * Si + 6.5 * Mn + 2.0 * Cr + 11.5 * Ni + 5.5 * Al + 500 * P + 700 * S
        )
        if C <= 0.09:
            return 1538 - (478 * C + base)
        elif 0.09 < C <= 0.17:
            return 1495 - base
        else:
            return 1527 - (187.5 * C + base)

    results["Solidus"]["1997_宝钢技术_平居公式"] = solidus_1997()

    # 2. 2006年《薄板坯连铸连轧钢的组织性能控制》
    results["Solidus"]["2006_薄板坯连铸连轧"] = 1537 - (
        175 * C + 20 * Si + 30 * Mn + 280 * P + 575 * S + 7.5 * Al + 160 * O
    )

    # 3. 2007年《品种钢、优特钢连铸900问》
    results["Solidus"]["2007_连铸900问"] = 1534 - 2.29 * (
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

    # 4. 2011年《连续铸钢生产技术》
    results["Solidus"]["2011_连续铸钢生产技术"] = 1471 - (
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

    # 5. 2012年《钢铁》第10期第28页
    results["Solidus"]["2012_钢铁"] = 1536 - (
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

    return results


def plot_temperature_comparison(results):
    """绘制横向柱状图对比不同公式的温度预测"""
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
    plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

    # 创建自定义布局 (1行2列)
    fig = plt.figure(figsize=(24, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    # 获取并过滤数据(确保只包含钢的数据)
    liquids = {k: v for k, v in results["Liquidus"].items() if "铸铁" not in k}
    solids = {k: v for k, v in results["Solidus"].items() if "铸铁" not in k}

    if not liquids:
        raise ValueError("没有有效的液相线数据")
    if not solids:
        raise ValueError("没有有效的固相线数据")

    # 计算平均值
    liquid_avg = sum(liquids.values()) / len(liquids)
    solid_avg = sum(solids.values()) / len(solids)

    # 液相线绘图(钢材) - 左上
    ax1 = plt.subplot(gs[0, 0])
    sorted_liquids = sorted(liquids.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_liquids]
    values = [x[1] for x in sorted_liquids]

    # 使用渐变颜色和细条形
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(values), max(values))
    colors = cmap(norm(values))

    plt.barh(names, values, height=0.6, color=colors)
    plt.title("钢的液相线温度比较", fontsize=16, pad=20)
    plt.xlabel("温度 (°C)", fontsize=14)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    # 添加平均值参考线
    plt.axvline(liquid_avg, color="red", linestyle="--", linewidth=2)
    plt.text(
        liquid_avg + 1,
        0.1,  # 调整到图表底部
        f"平均值: {liquid_avg:.1f}°C",
        color="blue",
        fontsize=15,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # 固相线绘图(钢材) - 右上
    ax2 = plt.subplot(gs[0, 1])
    sorted_solids = sorted(solids.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_solids]
    values = [x[1] for x in sorted_solids]

    # 使用渐变颜色和细条形
    cmap = plt.cm.plasma
    norm = plt.Normalize(min(values), max(values))
    colors = cmap(norm(values))

    plt.barh(names, values, height=0.6, color=colors)
    plt.title("钢的固相线温度比较", fontsize=16, pad=20)
    plt.xlabel("温度 (°C)", fontsize=14)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    # 添加平均值参考线
    plt.axvline(solid_avg, color="red", linestyle="--", linewidth=2)
    plt.text(
        solid_avg + 1,
        0.1,  # 调整到图表底部
        f"平均值: {solid_avg:.2f}°C",
        color="blue",
        fontsize=15,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # 设置坐标轴范围(自动调整)
    def get_axis_range(values, margin=0.1):
        min_val = min(values)
        max_val = max(values)
        span = max_val - min_val
        return min_val - span * margin, max_val + span * margin

    ax1.set_xlim(*get_axis_range(liquids.values()))
    ax2.set_xlim(*get_axis_range(solids.values()))

    plt.tight_layout()
    plt.show()


# -------------------------- 示例用法 --------------------------
if __name__ == "__main__":
    # 示例1：低碳钢
    composition1 = {
        "C": 0.1,
        "Si": 0.2,
        "Mn": 0.5,
        "P": 0.01,
        "S": 0.005,
        "Cr": 0.3,
        "Ni": 0.1,
        "Mo": 0.05,
    }
    results1 = compute_temperatures(composition1)
    plot_temperature_comparison(results1)

    # # 示例2：双相不锈钢
    # composition2 = {"C": 0.03, "Cr": 22, "Ni": 5, "Mo": 3, "N": 0.15}
    # results2 = compute_temperatures(composition2)
    # plot_temperature_comparison(results2)

    # # 示例3：高熵合金
    # composition3 = {"Fe": 20, "Co": 20, "Ni": 20, "Cr": 20, "Mn": 20}
    # results3 = compute_temperatures(composition3)
    # plot_temperature_comparison(results3)
