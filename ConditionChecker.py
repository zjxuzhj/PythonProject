from dataclasses import dataclass
from typing import Optional

import pandas as pd
from dataclasses import dataclass
from typing import Optional



@dataclass
class RuleEnumConfig:
    # --- 形态与趋势参数 ---
    MA_SPREAD_RATIO_THRESHOLD = 0.012  # MA5,10,20粘合的阈值
    MA_ZHONG_NIAN_HE_THRESHOLD = 0.025  # 中期均线粘合阈值
    MA_LONG_NIAN_HE_THRESHOLD = 0.02  # 长期均线粘合阈值

    PRE_LIMIT_UP_CHAOTIC_AMPLITUDE = 0.05  # 涨停前混乱形态的振幅阈值
    PRE_LIMIT_UP_CHAOTIC_BODY_RATIO = 0.3  # 涨停前混乱形态的实体比例阈值
    P1_HUGE_VOLUME_GREEN_CANDLE_RATIO = 3.6  # T+1放量阴线成交量倍数


class ConditionChecker:
    def __init__(self, df: pd.DataFrame, limit_up_day_idx: int, offset: int, stock_info: dict):
        """
        最终统一版初始化方法。
        :param df: DataFrame
        :param limit_up_day_idx: 首板日（锚点）的索引
        :param offset: 相对于首板日的偏移量 (0表示检查首板日当天, >0表示检查买入日)
        :param config: 策略配置
        :param stock_info: 股票信息
        """
        self.df = df
        self.limit_up_idx = limit_up_day_idx
        self.offset = offset
        # 我们当前正在分析的日子
        self.current_day_idx = limit_up_day_idx + offset
        self.config = RuleEnumConfig()
        self.stock_info = stock_info

        # --- 预加载所有可能用到的数据，全部相对于 limit_up_idx ---
        self.limit_up_day_data = self.df.iloc[self.limit_up_idx] if self.limit_up_idx < len(df) else None
        self.day_m1_data = self.df.iloc[self.limit_up_idx - 1] if self.limit_up_idx > 0 else None

        self.day_p1_data = self.df.iloc[self.limit_up_idx + 1] if self.limit_up_idx + 1 < len(df) else None
        self.day_p2_data = self.df.iloc[self.limit_up_idx + 2] if self.limit_up_idx + 2 < len(df) else None
        self.day_p3_data = self.df.iloc[self.limit_up_idx + 3] if self.limit_up_idx + 3 < len(df) else None

    # --- 这里是您提到的 "is_nian_he" 的具体实现 ---
    def is_51020_m1_nian_he(self, threshold: Optional[float] = None) -> bool:
        """
        检查T-1日短期均线(5,10,20)是否粘合
        """
        actual_threshold = threshold if threshold is not None else self.config.MA_SPREAD_RATIO_THRESHOLD
        if self.limit_up_idx == 0: return False  # 第一天无T-1数据

        ma5 = self.day_m1_data['ma5']
        ma10 = self.day_m1_data['ma10']
        ma20 = self.day_m1_data['ma20']

        ma_list = [ma5, ma10, ma20]
        if any(ma is None or pd.isna(ma) or ma == 0 for ma in ma_list):
            return False  # 数据不足

        max_ma = max(ma_list)
        min_ma = min(ma_list)
        avg_ma = sum(ma_list) / 3

        spread_ratio = (max_ma - min_ma) / avg_ma

        # 使用配置中的阈值进行判断
        return spread_ratio < actual_threshold


    def is_p1_huge_volume_green_candle(self) -> bool:
        """条件3：检查T+1是否为巨量阴线"""
        if self.limit_up_idx >= len(self.df) - 1: return True  # 没有T+1数据，直接排除

        volume_p0 = self.limit_up_day_data['volume']
        volume_p1 = self.day_p1_data['volume']
        is_red_candle_p1 = self.day_p1_data['close'] >= self.day_p1_data['open']

        volume_check = volume_p1 >= volume_p0 * self.config.P1_HUGE_VOLUME_GREEN_CANDLE_RATIO

        return volume_check and not is_red_candle_p1
