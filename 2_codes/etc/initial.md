# 建立计算模型及初始化的功能

建立计算模型及初始化的目的，是为了给Iterator.py直接提供一系列的输出参数，而不需要每次都重新计算。

## 输入的参数及来源

    - mesh_size，空间步长
    - time_step，时间步长
    - 分段信息，位于2_codes\results\process_data.json文件内

## 提供的参数内容

提供参数的内容包括：
    - Lx,slab_width/2/1000
    - Ly,slab_thickness/2/1000
    - h_top,
    - h_right,
    - T_inf_top,
    - T_inf_right,
    - dt,
    - total_time,
    - initial_temp,  # 二维初始温度场 [K]
    - tol=1e-6,  # 使用全局定义的容差

## 具体的计算过程

1. 获取数据输入：从参数输入来源，获取到所有信息；
2. 整理数据，建立数据输出模型：将数据输入处理整合，分成多段输入，让计算模块只需要读取生成的initialize.json文件，不需要再次计算；如
   1. 第1段，结晶器段，Lx，Ly，h_top，h_right，T_inf_top，T_inf_right，dt，total_time，tol；其中的initial_temp，是连铸温度的均匀分布温度场
   2. 第2段，水冷区第一段，Lx，Ly，h_top，h_right，T_inf_top，T_inf_right，dt，total_time，tol；其中的initial_temp，(第1段结晶器出口的温度场，需要计算，暂不输入)
   3. ……
   4. 第i段，水冷区第i-1段，Lx，Ly，h_top，h_right，T_inf_top，T_inf_right，dt，total_time，tol；其中的initial_temp，(第i-1段水冷出口的温度场，需要计算，暂不输入)
   5. ……
   6. 第j段，水冷区第最后1段，Lx，Ly，h_top，h_right，T_inf_top，T_inf_right，dt，total_time，tol；（其中的initial_temp，第j-1段水冷出口的温度场，暂不输入）
   7. 最后一段，空冷段，Lx，Ly，h_top，h_right，T_inf_top，T_inf_right，dt，total_time，tol；（其中的initial_temp，最后一段水冷段j段出口的温度场，需要计算，暂不输入）
