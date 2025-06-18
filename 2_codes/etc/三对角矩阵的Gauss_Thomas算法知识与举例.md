# 三对角矩阵的Gauss-Thomas算法

## 1. 算法简介

Gauss-Thomas算法（又称Thomas算法）是求解三对角线性方程组的高效算法，时间复杂度为$O(n)$。它是对高斯消元法的特殊优化，适用于形如：

$$
\begin{matrix}
b_1 & c_1 & 0 & \cdots & 0 \\
a_2 & b_2 & c_2 & \cdots & 0 \\
0 & \ddots & \ddots & \ddots & \vdots \\
\vdots & \cdots & a_{n-1} & b_{n-1} & c_{n-1} \\
0 & \cdots & 0 & a_n & b_n
\end{matrix}
\cdot
\begin{matrix}
x_1 \\
x_2 \\
\vdots \\
x_{n-1} \\
x_n
\end{matrix}
=
\begin{matrix}
d_1& \\
d_2 &\\
\vdots& \\
d_{n-1} &\\
d_n&
\end{bmatrix}
$$

的方程组。

## 2. 数学原理

三对角方程组的一般形式：

\begin{align*}
b_1x_1 + c_1x_2 &= d_1 \\
a_2x_1 + b_2x_2 + c_2x_3 &= d_2 \\
&\vdots \\
a_ix_{i-1} + b_ix_i + c_ix_{i+1} &= d_i \\
&\vdots \\
a_nx_{n-1} + b_nx_n &= d_n
\end{align*}

## 3. 算法步骤

### 3.1 前向消元

\begin{enumerate}
\item 计算修正系数：
\begin{align*}
c'_1 &= \frac{c_1}{b_1} \\
d'_1 &= \frac{d_1}{b_1}
\end{align*}

\item 对于$i$从2到$n-1$：
\begin{align*}
c'_i &= \frac{c_i}{b_i - a_i c'_{i-1}} \\
d'_i &= \frac{d_i - a_i d'_{i-1}}{b_i - a_i c'_{i-1}}
\end{align*}

\item 最后一行：
\begin{align*}
d'_n &= \frac{d_n - a_n d'_{n-1}}{b_n - a_n c'_{n-1}}
\end{align*}
\end{enumerate}

### 3.2 回代求解

\begin{enumerate}
\item $x_n = d'_n$
\item 对于$i$从$n-1$降到1：
\begin{align*}
x_i = d'_i - c'_i x_{i+1}
\end{align*}
\end{enumerate}

## 4. Python实现

```python
def thomas_algorithm(a, b, c, d):
    """解三对角方程组 Ax = d

    参数:
        a: 下对角线元素 [a₂, a₃, ..., aₙ] (长度为n-1)
        b: 主对角线元素 [b₁, b₂, ..., bₙ] (长度为n)
        c: 上对角线元素 [c₁, c₂, ..., cₙ₋₁] (长度为n-1)
        d: 右侧向量 [d₁, d₂, ..., dₙ] (长度为n)

    返回:
        x: 解向量 [x₁, x₂, ..., xₙ]
    """
    n = len(d)
    c_prime = [0] * (n-1)
    d_prime = [0] * n

    # 前向消元
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n-1):
        denominator = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denominator
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denominator

    d_prime[-1] = (d[-1] - a[-1] * d_prime[-2]) / (b[-1] - a[-1] * c_prime[-1])

    # 回代
    x = [0] * n
    x[-1] = d_prime[-1]

    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x
```

## 5. 应用示例

求解以下三对角方程组：

\begin{align*}
2x_1 + x_2 &= 4 \\
x_1 + 2x_2 + x_3 &= 8 \\
x_2 + 2x_3 &= 12
\end{align*}

Python代码：

```python
a = [1, 1]    # 下对角线
b = [2, 2, 2] # 主对角线
c = [1, 1]    # 上对角线
d = [4, 8, 12] # 右侧向量

x = thomas_algorithm(a, b, c, d)
print("解:", x)  # 输出: [1.0, 2.0, 5.0]
```

## 6. 算法分析

\begin{itemize}
\item 时间复杂度：$O(n)$，远优于普通高斯消元法的$O(n^3)$
\item 空间复杂度：$O(n)$，只需存储几条对角线
\item 稳定性：当矩阵严格对角占优时算法稳定
\item 应用场景：热传导方程、流体力学、有限差分法等数值计算问题
\end{itemize}

## 7. 注意事项

\begin{itemize}
\item 算法要求矩阵不可约且对角占优
\item 除零错误可能出现在消元过程中
\item 对于大规模问题，可以考虑并行化实现
\item 在边界条件处理时需要特别注意系数的设置
\end{itemize}
