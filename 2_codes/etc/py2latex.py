from sympy import latex, parse_expr, Symbol
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    split_symbols,
)
import re


def expr_to_latex(expr_str: str) -> str:
    """
    将数学表达式转换为 LaTeX，保留小数次方并正确处理科学计数法

    参数:
        expr_str (str): 输入表达式（如 "P = 1.23e5*t**2.5 + sqrt(x)"）

    返回:
        str: LaTeX 公式

    示例:
        >>> expr_to_latex("sqrt(x) + x**0.5")
        '\\sqrt{x} + x^{0.5}'
    """
    # 预处理步骤
    expr_str = expr_str.replace("^", "**")  # 统一幂运算符

    # 将科学计数法转换为 a*10^b 形式（保留符号）
    expr_str = re.sub(
        r"(\d+\.?\d*)[eE]([+-]?\d+)",
        lambda m: f"{m.group(1)}*10**{m.group(2)}",
        expr_str,
    )

    # 配置解析规则（禁止自动计算 10**5）
    transformations = standard_transformations + (
        implicit_multiplication_application,
        split_symbols,
        convert_xor,
    )

    try:
        # 解析表达式
        if "=" in expr_str:
            lhs, rhs = expr_str.split("=", 1)
            expr = parse_expr(
                rhs.strip(), transformations=transformations, evaluate=False
            )
            lhs = lhs.strip()
        else:
            expr = parse_expr(expr_str, transformations=transformations, evaluate=False)
            lhs = ""

        # 生成原始 LaTeX 代码（禁止根号转换）
        latex_code = latex(expr, mul_symbol="\\cdot", root_notation=False)

        # 后处理：将 10**5 转换为 10^{5}
        latex_code = re.sub(r"10\*\*(\d+)", r"10^{\1}", latex_code)

        # 将乘号与 10^{n} 结合成 \cdot 10^{n}
        latex_code = re.sub(r"\\cdot\s*10\^{", r" \\cdot 10^{", latex_code)

        return f"{lhs} = {latex_code}" if lhs else latex_code

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------------- 测试用例 ----------------------
if __name__ == "__main__":
    test_cases = [
        "x**0.5 + y**0.333",
        "h = V**0.451*(1 - 0.0075*T_w)",
        "y = (A**0.25 + B**2.3)/C",
        "P = 1.23e5*t**2.5",
        "sqrt(x) + x**0.5",
        "1.23e-4 * 10**3",
    ]

    for expr in test_cases:
        print(f"输入: {expr}")
        print(f"输出: {expr_to_latex(expr)}\n")
