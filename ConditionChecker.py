from dataclasses import dataclass
from typing import Optional

import pandas as pd
from dataclasses import dataclass
from typing import Optional



@dataclass
class RuleEnumConfig:
    # --- 形态与趋势参数 ---
    MA_SPREAD_RATIO_THRESHOLD = 0.012  # MA5,10,20粘合的阈值
    MA_ZHONG_NIAN_HE_THRESHOLD = 0.015  # 中期均线粘合阈值
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

    def is_ma_cohesive(self, ma_list_names: list[str], threshold: float, day_data: pd.Series = None) -> bool:
        """
        一个通用的均线粘合检查方法。
        :param ma_list_names: 需要检查的均线列名列表, e.g., ['ma5', 'ma10', 'ma20']
        :param threshold: 判断粘合的离散度阈值
        :param day_data: 可选参数，指定要检查哪一天的数据。默认为None，表示检查锚点日(首板日)的前一天(T-1)。
        :return: 如果粘合则返回 True，否则 False。
        """
        # 如果未指定日期，默认使用 T-1 的数据，这是最常见的场景
        if day_data is None:
            if self.day_m1_data is None:
                return False  # 没有T-1数据，无法判断
            day_data_to_check = self.day_m1_data
        else:
            day_data_to_check = day_data
        ma_values = [day_data_to_check.get(ma_name) for ma_name in ma_list_names]
        if any(pd.isna(val) or val is None or val == 0 for val in ma_values):
            return False
        max_ma = max(ma_values)
        min_ma = min(ma_values)
        avg_ma = sum(ma_values) / len(ma_values)
        if avg_ma > 0:
            spread_ratio = (max_ma - min_ma) / avg_ma
            return spread_ratio < threshold
        return False

    def is_51020_m1_nian_he(self, threshold: Optional[float] = None) -> bool:
        # 检查T-1日短期均线(5,10,20)是否粘合
        actual_threshold = threshold if threshold is not None else self.config.MA_SPREAD_RATIO_THRESHOLD
        return self.is_ma_cohesive(['ma5', 'ma10', 'ma20'], actual_threshold)

    def is_102030_m1_nian_he(self, threshold: Optional[float] = None) -> bool:
        actual_threshold = threshold if threshold is not None else 0.025
        return self.is_ma_cohesive(['ma10', 'ma20', 'ma30'], actual_threshold)

    def is_203055_m1_nian_he(self, threshold: Optional[float] = None) -> bool:
        actual_threshold = threshold if threshold is not None else self.config.MA_ZHONG_NIAN_HE_THRESHOLD
        return self.is_ma_cohesive(['ma20', 'ma30', 'ma55'], actual_threshold)

    def is_55120250_m1_nian_he(self, threshold: Optional[float] = None) -> bool:
        actual_threshold = threshold if threshold is not None else self.config.MA_LONG_NIAN_HE_THRESHOLD
        return self.is_ma_cohesive(['ma55', 'ma120', 'ma250'], actual_threshold)

    def is_p1_huge_volume_green_candle(self) -> bool:
        """条件3：检查T+1是否为巨量阴线"""
        if self.limit_up_idx >= len(self.df) - 1: return True  # 没有T+1数据，直接排除

        volume_p0 = self.limit_up_day_data['volume']
        volume_p1 = self.day_p1_data['volume']
        is_red_candle_p1 = self.day_p1_data['close'] >= self.day_p1_data['open']

        volume_check = volume_p1 >= volume_p0 * self.config.P1_HUGE_VOLUME_GREEN_CANDLE_RATIO

        return volume_check and not is_red_candle_p1
