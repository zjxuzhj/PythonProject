from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StockInfo:
    """用于存储股票基本信息的标准对象"""
    code: str # code可能存在两种格式，如 600000.SH 或 sh600000
    name: str
    market_value: float
    theme: str  # 假设一个股票可以有多个题材，我们用字符串列表

    # 预留未来可能添加的字段，使用 Optional 表示它可能不存在
    industry: Optional[str] = None

    def __str__(self):
        """定义一个更友好的打印格式"""
        theme_str = ', '.join(self.theme) if self.theme else '无'
        industry_str = self.industry if self.industry else '未指定'
        return (f"股票信息(代码: {self.code}, 名称: {self.name}, "
                f"市值: {self.market_value:.2f}亿, 题材: {theme_str}, "
                f"行业: {industry_str})")