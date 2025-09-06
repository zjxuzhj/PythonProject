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

    MA_SUPPORT_RECOVERY_TOLERANCE = 0.05  # 均线支撑回升形态的容差 (5%)

    PLATFORM_SPREAD_THRESHOLD_STRICT = 0.01  # 平台波动阈值(严格, 1%)
    PLATFORM_SPREAD_THRESHOLD_LOOSE = 0.02   # 平台波动阈值(宽松, 2%)

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

    def is_10_m1m2m3_support_only_close(self) -> bool:
        days_to_check_indices = [
            self.limit_up_idx - 1,
            self.limit_up_idx - 2,
            self.limit_up_idx - 3
        ]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma10', only_close=True)

    def is_10_p1p2p3_support(self) -> bool:
        days_to_check_indices = [
            self.limit_up_idx + 1,
            self.limit_up_idx + 2,
            self.limit_up_idx + 3
        ]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma10')

    def is_20_p0m1m2_support(self) -> bool:
        days_to_check_indices = [
            self.limit_up_idx,
            self.limit_up_idx - 1,
            self.limit_up_idx - 2
        ]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma20')

    def is_20_p1p2p3_support(self) -> bool:
        days_to_check_indices = [self.limit_up_idx + 1, self.limit_up_idx + 2, self.limit_up_idx + 3]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma20')

    def is_20_p1p2p3_support_only_close(self) -> bool:
        days_to_check_indices = [self.limit_up_idx + 1, self.limit_up_idx + 2, self.limit_up_idx + 3]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma20', only_close=True)

    def is_30_m1m2m3_support_only_close(self) -> bool:
        days_to_check_indices = [self.limit_up_idx - 1, self.limit_up_idx - 2, self.limit_up_idx - 3]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma30', only_close=True)

    def is_30_p0m1m2_support(self) -> bool:
        days_to_check_indices = [
            self.limit_up_idx,
            self.limit_up_idx - 1,
            self.limit_up_idx - 2
        ]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma30')

    def is_30_p1p2p3_support(self) -> bool:
        days_to_check_indices = [
            self.limit_up_idx + 1,
            self.limit_up_idx + 2,
            self.limit_up_idx + 3
        ]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma30')

    def is_30_p1p2p3_support_only_close(self) -> bool:
        days_to_check_indices = [self.limit_up_idx + 1, self.limit_up_idx + 2, self.limit_up_idx + 3]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma30', only_close=True)

    def is_60_m1m2m3_support(self) -> bool:
        days_to_check_indices = [
            self.limit_up_idx - 1,
            self.limit_up_idx - 2,
            self.limit_up_idx - 3
        ]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma60')

    def is_60_p0m1m2_support(self) -> bool:
        days_to_check_indices = [
            self.limit_up_idx,
            self.limit_up_idx - 1,
            self.limit_up_idx - 2
        ]
        return self.is_persistently_supported_by_ma(day_indices=days_to_check_indices, ma_name='ma60')

    def is_120_p0m1m2_support(self) -> bool:
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
            above_ma_threshold: float = 0.98,
            only_close: bool = False
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
            day_close = day_data['close']
            is_above_ma = day_open >= ma_value * above_ma_threshold
            if only_close:
                is_close_to_ma = (
                        (abs(day_close - ma_value) / ma_value) <= proximity_threshold
                )
            else:
                is_close_to_ma = (
                        (abs(day_open - ma_value) / ma_value) <= proximity_threshold or
                        (abs(day_low - ma_value) / ma_value) <= proximity_threshold or
                        (abs(day_close - ma_value) / ma_value) <= proximity_threshold
                )
            if not (is_above_ma and is_close_to_ma):
                return False
        return True

    def is_10_m1_support_005(self) -> bool:
        return self.is_ma_pierce_and_recover(
            ma_name='ma10',
            day_data=self.day_m1_data,
            tolerance=self.config.MA_SUPPORT_RECOVERY_TOLERANCE
        )

    def is_30_m1_support(self) -> bool:
        return self.is_ma_pierce_and_recover(
            ma_name='ma30',
            day_data=self.day_m1_data
        )
    def is_ma_pierce_and_recover(
            self,
            ma_name: str,
            day_data: pd.Series,
            tolerance: float = 0.0
    ) -> bool:
        """
        检查在指定的某一天，价格是否形成下探均线后又回升的支撑形态。
        精确匹配逻辑: low < ma * (1 + tolerance) <= close

        :param ma_name: 要检查的均线列名 (e.g., 'ma10')。
        :param day_data: 要检查的当日的DataFrame行数据 (e.g., self.day_m1_data)。
        :param tolerance: 均线值的容差，0.05代表5%。
        :return: 如果形成该形态，则返回 True，否则 False。
        """
        # 检查传入的数据是否有效
        if day_data is None:
            return False
        ma_value = day_data.get(ma_name)
        day_low = day_data['low']
        day_close = day_data['close']
        # 检查数值有效性
        if pd.isna(ma_value) or pd.isna(day_low) or pd.isna(day_close) or ma_value <= 0:
            return False
        # 核心逻辑判断
        ma_with_tolerance = ma_value * (1 + tolerance)
        return day_low < ma_with_tolerance <= day_close

    def is_stable_platform(self) -> bool:
        support_level = self._find_stable_platform_level(
            start_offset_from_anchor=-4,
            end_offset_from_anchor=0,
            spread_threshold=self.config.PLATFORM_SPREAD_THRESHOLD_STRICT
        )
        return support_level is not None

    def is_stable_platform_to_new_low(self) -> bool:
        return self._is_platform_breakdown_to_new_low(
            platform_start_offset=-4,
            platform_end_offset=-1,
            spread_threshold=self.config.PLATFORM_SPREAD_THRESHOLD_STRICT
        )

    def _find_stable_platform_level(
            self,
            start_offset_from_anchor: int,
            end_offset_from_anchor: int,
            spread_threshold: float
    ) -> Optional[float]:
        """
        [内部辅助方法] 在指定窗口内查找是否存在一个稳定的价格平台。

        :param start_offset_from_anchor: 平台窗口开始位置的偏移量 (e.g., -5)。
        :param end_offset_from_anchor: 平台窗口结束位置的偏移量 (e.g., -1)。
        :param spread_threshold: 平台价格波动的容忍阈值。
        :return: 如果找到稳定平台，则返回平台的平均支撑位；否则返回 None。
        """
        platform_window = self.df.iloc[
                          self.limit_up_idx + start_offset_from_anchor: self.limit_up_idx + end_offset_from_anchor]
        if len(platform_window) != (end_offset_from_anchor - start_offset_from_anchor):
            return None  # 数据窗口长度不足
        bottom_prices = []
        for _, day in platform_window.iterrows():
            bottom_prices.append(day['open'] if day['close'] >= day['open'] else day['close'])
        if not all(p > 0 for p in bottom_prices):
            return None
        avg_price = sum(bottom_prices) / len(bottom_prices)
        if avg_price > 0 and (max(bottom_prices) - min(bottom_prices)) / avg_price < spread_threshold:
            return avg_price  # 找到了平台，返回支撑位
        return None

    def _is_platform_breakdown_to_new_low(
            self,
            platform_start_offset: int,
            platform_end_offset: int,
            spread_threshold: float
    ) -> bool:
        """
        检查是否存在一个稳定平台，并且在平台之后的一天(T-1)破位创下新低。

        :param platform_start_offset: 平台窗口开始位置的偏移量 (e.g., -5)。
        :param platform_end_offset: 平台窗口结束位置的偏移量 (e.g., -2)。
        :param spread_threshold: 平台价格波动的容忍阈值。
        :return: 如果该模式成立，返回 True。
        """
        # 1. 查找是否存在稳定平台
        support_line_level = self._find_stable_platform_level(
            platform_start_offset, platform_end_offset, spread_threshold
        )
        if support_line_level is None:
            return False  # 没有找到平台，模式不成立
        # 2. 检查T-1日是否收盘破位
        if self.day_m1_data is None or self.day_m1_data['close'] >= support_line_level:
            return False  # 没有T-1数据或没有破位，模式不成立
        # 3. 检查T-1日的最低价是否为近期新低
        entire_window = self.df.iloc[self.limit_up_idx + platform_start_offset: self.limit_up_idx]  # e.g., T-5 到 T-1
        if self.day_m1_data['low'] == entire_window['low'].min():
            return True  # 破位且创下新低，模式成立
        return False

    def is_p1_huge_volume_green_candle(self) -> bool:
        """条件3：检查T+1是否为巨量阴线"""
        if self.limit_up_idx >= len(self.df) - 1: return True  # 没有T+1数据，直接排除

        volume_p0 = self.limit_up_day_data['volume']
        volume_p1 = self.day_p1_data['volume']
        is_red_candle_p1 = self.day_p1_data['close'] >= self.day_p1_data['open']

        volume_check = volume_p1 >= volume_p0 * self.config.P1_HUGE_VOLUME_GREEN_CANDLE_RATIO

        return volume_check and not is_red_candle_p1
