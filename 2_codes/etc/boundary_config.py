import json
from typing import List, Dict, Union


class BoundaryConfig:
    """边界条件分段配置管理类"""

    def __init__(self, total_time: float = 300.0):
        """初始化边界条件配置

        Args:
            total_time (float): 总模拟时间(s)，默认为300秒
        """
        self.total_time = total_time
        self.segments: List[Dict[str, Union[str, float]]] = []

    def add_segment(
        self, start: float, end: float, boundary_type: str, **kwargs
    ) -> None:
        """添加边界条件分段

        Args:
            start (float): 分段开始时间(s)
            end (float): 分段结束时间(s)
            boundary_type (str): 边界类型，支持"first_kind"(第一类)、
                               "second_kind"(第二类)、"third_kind"(第三类)
            **kwargs: 边界条件参数，根据类型不同需要不同参数:
                - 第一类边界: temperature=温度值(℃)
                - 第二类边界: q_top=上边界热流密度(W/m²), q_right=右边界热流密度(W/m²)
                - 第三类边界: h_top=上边界换热系数(W/m²·K), h_right=右边界换热系数(W/m²·K),
                            T_inf_top=上边界环境温度(℃), T_inf_right=右边界环境温度(℃)
        """
        if start >= end:
            raise ValueError("开始时间必须小于结束时间")

        if boundary_type not in ["first_kind", "second_kind", "third_kind"]:
            raise ValueError("不支持的边界类型")

        segment = {"start": start, "end": end, "type": boundary_type, **kwargs}
        self.segments.append(segment)

    def remove_segment(self, index: int) -> None:
        """删除指定索引的分段"""
        if 0 <= index < len(self.segments):
            self.segments.pop(index)

    def update_segment(self, index: int, **kwargs) -> None:
        """更新指定索引的分段参数"""
        if 0 <= index < len(self.segments):
            self.segments[index].update(kwargs)

    def validate(self) -> bool:
        """验证分段配置是否有效"""
        if not self.segments:
            return False

        # 检查时间连续性
        prev_end = 0.0
        for seg in sorted(self.segments, key=lambda x: x["start"]):
            if seg["start"] != prev_end:
                return False
            prev_end = seg["end"]

        # 检查是否覆盖整个时间段
        if prev_end != self.total_time:
            return False

        return True

    def to_json(self, file_path: str) -> None:
        """将配置保存为JSON文件

        Args:
            file_path (str): JSON文件路径
        """
        config = {"total_time": self.total_time, "segments": self.segments}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    @classmethod
    def from_json(cls, file_path: str) -> "BoundaryConfig":
        """从JSON文件加载配置

        Args:
            file_path (str): JSON文件路径

        Returns:
            BoundaryConfig: 加载后的配置对象
        """
        with open(file_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        bc = cls(config["total_time"])
        bc.segments = config["segments"]
        return bc

    def get_segments(self) -> List[Dict[str, Union[str, float]]]:
        """获取边界条件分段列表"""
        return self.segments


# 示例用法
if __name__ == "__main__":
    # 创建配置对象
    config = BoundaryConfig(total_time=180)

    # 添加分段
    config.add_segment(
        0, 60, "third_kind", h_top=1000, h_right=1000, T_inf_top=30, T_inf_right=30
    )

    config.add_segment(60, 120, "second_kind", q_top=-50000, q_right=-50000)

    config.add_segment(120, 180, "second_kind", q_top=-30000, q_right=-30000)

    # 保存到文件
    config.to_json("boundary_config.json")

    # 从文件加载
    loaded_config = BoundaryConfig.from_json("boundary_config.json")
    print(loaded_config.get_segments())
