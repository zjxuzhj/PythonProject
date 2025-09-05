from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class RuleEnumConfig:
    # --- 形态与趋势参数 ---
    MA_SPREAD_RATIO_THRESHOLD = 0.012  # MA5,10,20粘合的阈值
    MA_ZHONG_NIAN_HE_THRESHOLD = 0.015  # 中期均线粘合阈值
    MA_LONG_NIAN_HE_THRESHOLD = 0.02  # 长期均线粘合阈值

    PRE_LIMIT_UP_CHAOTIC_AMPLITUDE = 0.05  # 涨停前混乱形态的振幅阈值
    PRE_LIMIT_UP_CHAOTIC_BODY_RATIO = 0.3  # 涨停前混乱形态的实体比例阈值
    P1_HUGE_VOLUME_GREEN_CANDLE_RATIO = 3.6  # T+1放量阴线成交量倍数

    MA10_MA55_COHESION_THRESHOLD = 0.01  # 10日线和55日线粘合阈值


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

    def is_10120_m1_nian_he(self, threshold: Optional[float] = None) -> bool:
        actual_threshold = threshold if threshold is not None else 0.01
        return self.is_ma_cohesive(['ma10', 'ma120'], actual_threshold)

    def is_55120_m1_nian_he(self, threshold: Optional[float] = None) -> bool:
        actual_threshold = threshold if threshold is not None else 0.005
        return self.is_ma_cohesive(['ma55', 'ma120'], actual_threshold)

    def is_55250_m1_nian_he(self, threshold: Optional[float] = None) -> bool:
        actual_threshold = threshold if threshold is not None else 0.012
        return self.is_ma_cohesive(['ma55', 'ma250'], actual_threshold)

    def is_555_m1_abs_nian_he(self, threshold: Optional[float] = None) -> bool:
        # 检查T-1日短期均线(5,55)是否粘合，取绝对值
        actual_threshold = threshold if threshold is not None else self.config.MA10_MA55_COHESION_THRESHOLD
        return self.are_mas_within_percent_diff('ma5', 'ma55', actual_threshold)

    def are_mas_within_percent_diff(self, base_ma_name: str, other_ma_name: str, threshold: float,
                                    day_data: pd.Series = None) -> bool:
        """
        检查一个均线(other_ma)是否在另一个基准均线(base_ma)的特定百分比范围内。
        这精确匹配了 abs(A - B) / A < threshold 的逻辑。

        :param base_ma_name: 作为分母的基准均线列名 (e.g., 'ma5')
        :param other_ma_name: 另一个要比较的均线列名 (e.g., 'ma55')
        :param threshold: 百分比差异的阈值
        :param day_data: 可选参数，指定要检查哪一天的数据。默认为None，表示检查T-1。
        :return: 如果在范围内则返回 True，否则 False。
        """
        if day_data is None:
            if self.day_m1_data is None:
                return False
            day_data_to_check = self.day_m1_data
        else:
            day_data_to_check = day_data
        base_ma_val = day_data_to_check.get(base_ma_name)
        other_ma_val = day_data_to_check.get(other_ma_name)
        if pd.isna(base_ma_val) or pd.isna(
                other_ma_val) or base_ma_val is None or other_ma_val is None or base_ma_val == 0:
            return False
        percent_diff = abs(base_ma_val - other_ma_val) / base_ma_val
        return percent_diff < threshold

    def is_persistently_20_supported_p0m1m2(self) -> bool:
        days_to_check_indices = [
            self.limit_up_idx,
            self.limit_up_idx - 1,
            self.limit_up_idx - 2
        ]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma20')

    def is_persistently_20_supported_p1p2p3(self) -> bool:
        days_to_check_indices = [
            self.limit_up_idx + 1,
            self.limit_up_idx + 2,
            self.limit_up_idx + 3
        ]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma20')

    def is_persistently_60_supported_p0m1m2(self) -> bool:
        days_to_check_indices = [
            self.limit_up_idx,
            self.limit_up_idx - 1,
            self.limit_up_idx - 2
        ]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma60')

    def is_persistently_120_supported_p0m1m2(self) -> bool:
        days_to_check_indices = [
            self.limit_up_idx,
            self.limit_up_idx - 1,
            self.limit_up_idx - 2
        ]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma120')

    def is_persistently_supported_by_ma(
            self,
            day_indices: list[int],
            ma_name: str,
            proximity_threshold: float = 0.02,
            above_ma_threshold: float = 0.98
    ) -> bool:
        """
        检查在一系列指定的日子里，价格是否持续受到某条特定均线的支撑。

        :param day_indices: 一个包含待检查日期索引的列表。
        :param ma_name: 移动平均线的列名 (例如 'ma20', 'ma30')。
        :param proximity_threshold: 开盘价/最低价可以离均线有多近的容差。
        :param above_ma_threshold: 确保开盘价在均线上方的容差。
        :return: 如果在所有指定日期都受到支撑，则返回 True，否则返回 False。
        """
        for day_idx in day_indices:
            # 边界及数据有效性检查
            if day_idx < 0 or day_idx >= len(self.df):
                return False
            day_data = self.df.iloc[day_idx]
            ma_value = day_data.get(ma_name)
            if pd.isna(ma_value) or ma_value is None or ma_value <= 0:
                return False
            day_open = day_data['open']
            day_low = day_data['low']
            is_above_ma = day_open >= ma_value * above_ma_threshold
            is_close_to_ma = (
                    (abs(day_open - ma_value) / ma_value) <= proximity_threshold or
                    (abs(day_low - ma_value) / ma_value) <= proximity_threshold
            )
            if not (is_above_ma and is_close_to_ma):
                return False
        return True

    def is_p1_huge_volume_green_candle(self) -> bool:
        """条件3：检查T+1是否为巨量阴线"""
        if self.limit_up_idx >= len(self.df) - 1: return True  # 没有T+1数据，直接排除

        volume_p0 = self.limit_up_day_data['volume']
        volume_p1 = self.day_p1_data['volume']
        is_red_candle_p1 = self.day_p1_data['close'] >= self.day_p1_data['open']

        volume_check = volume_p1 >= volume_p0 * self.config.P1_HUGE_VOLUME_GREEN_CANDLE_RATIO

        return volume_check and not is_red_candle_p1
