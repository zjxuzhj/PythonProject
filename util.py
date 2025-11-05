from datetime import datetime, timedelta
import os
import requests
import re
import json
import logging
import pandas as pd
import xtquant.xtdata as xtdata

def get_yesterday(date_str):
    # 将输入的字符串转换为日期对象
    date_obj = datetime.strptime(date_str, "%Y%m%d")

    # 计算昨天的日期
    yesterday = date_obj - timedelta(days=1)

    # 将结果格式化为YYYYMMDD
    return yesterday.strftime("%Y%m%d")

def get_yesterdayNew(date_str):
    # 将输入的字符串转换为日期对象
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    # 计算昨天的日期
    yesterday = date_obj - timedelta(days=1)

    # 将结果格式化为YYYYMMDD
    return yesterday.strftime("%Y-%m-%d")

# 假设已定义节假日列表（需根据实际情况维护或对接交易所API）
holidays = [
    "2025-01-01",  # 元旦
    "2025-02-10",  # 春节
    "2025-04-04",  # 清明节
    "2025-05-01",  # 劳动节
    "2025-10-01",  # 国庆节
    # ... 其他节假日
]

def get_previous_trading_day(today_str, exchange="SSE"):
    """
    获取指定日期的上一个交易日（跳过周末和节假日）

    Args:
        today_str (str): 当前日期，格式 "YYYY-MM-DD"
        exchange (str): 交易所标识（默认为上交所 SSE）

    Returns:
        str: 上一个交易日的日期字符串，格式 "YYYY-MM-DD"
    """
    # 转换为 datetime 对象
    current_date = datetime.strptime(today_str, "%Y-%m-%d").date()

    # 循环查找上一个交易日
    max_retry = 30  # 防止无限循环（如连续假期）
    for _ in range(max_retry):
        current_date -= timedelta(days=1)

        # 1. 跳过周末（周一=0, 周日=6）
        if current_date.weekday() in [5, 6]:  # 周六、周日
            continue

        # 2. 跳过节假日
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str in holidays:
            continue

        # 如果是交易日，返回日期
        return date_str

    # 超出最大重试仍未找到（理论上不会触发）
    raise ValueError(f"无法在 {max_retry} 天内找到上一个交易日")

# 计算增长率
def calculate_growth_rate(today_num, yesterday_num):
    if yesterday_num == 0:  # 避免除零错误
        return 0.00
    growth_rate = ((today_num - yesterday_num) / yesterday_num) * 100
    return round(growth_rate, 2)  # 保留两位小数


class TradingCalendarUtil:
    """
    交易日历工具（基于miniQMT本地数据）

    - 判断逻辑：查询沪深300（基准：`000300.SH`）在指定日期是否有日线数据；若有则为交易日
    - 数据接口：`xtquant.xtdata.get_local_data`（miniQMT的本地数据接口）
    - 兼容性：与 `qmt_real_time_backtest.py` 中的基准证券与判断逻辑保持一致
    - 性能优化：内存缓存 + 可选本地JSON缓存，避免重复查询

    使用示例：
        cal = TradingCalendarUtil()
        cal.is_trading_day("20251009")  # True
        cal.is_trading_day("20251011")  # False（周六/节假日）
    """

    BENCHMARK_TICKER = '000300.SH'
    CACHE_FILE = os.path.join('back_test_data_cache', 'trading_calendar_cache.json')

    def __init__(self):
        self._cache = {}
        self._logger = logging.getLogger('TradingCalendarUtil')
        if not self._logger.handlers:
            logging.basicConfig(level=logging.INFO)
        self._ensure_cache_dir()
        self._load_cache()

    def _ensure_cache_dir(self):
        cache_dir = os.path.dirname(self.CACHE_FILE)
        if cache_dir and not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except Exception as e:
                # 目录创建失败不应阻塞主流程
                self._logger.warning(f"创建缓存目录失败: {e}")

    def _load_cache(self):
        try:
            if os.path.exists(self.CACHE_FILE):
                with open(self.CACHE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # 仅保留形如YYYYMMDD的key
                        self._cache = {k: bool(v) for k, v in data.items() if self._is_yyyymmdd(k)}
        except Exception as e:
            self._logger.warning(f"加载交易日缓存失败: {e}")

    def _save_cache(self):
        try:
            with open(self.CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False)
        except Exception as e:
            self._logger.warning(f"保存交易日缓存失败: {e}")

    def _is_yyyymmdd(self, s: str) -> bool:
        return isinstance(s, str) and len(s) == 8 and s.isdigit()

    def _normalize_date(self, date_str: str) -> str:
        """将日期字符串规范为YYYYMMDD格式，支持YYYY-MM-DD输入。"""
        if self._is_yyyymmdd(date_str):
            return date_str
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%Y%m%d")
        except Exception:
            # 无法解析，原样返回以便后续错误处理
            return date_str

    def is_trading_day(self, date_str: str) -> bool:
        """
        判断指定日期是否为交易日。

        参数：
        - date_str: 日期字符串，支持 `YYYYMMDD` 或 `YYYY-MM-DD`

        逻辑：
        - 快速路径：缓存命中直接返回
        - 周末短路：`weekday >= 5` 直接判定非交易日（与实时回测一致的行为）
        - 核心判断：使用 miniQMT 的 `get_local_data` 查询沪深300当日日线数据是否存在

        返回：
        - True 表示为交易日；False 表示非交易日或查询异常（记录日志）
        """
        norm = self._normalize_date(date_str)
        if not self._is_yyyymmdd(norm):
            self._logger.error(f"日期格式非法: {date_str}")
            return False

        if norm in self._cache:
            return self._cache[norm]

        try:
            dt = datetime.strptime(norm, "%Y%m%d")
        except Exception:
            self._logger.error(f"日期解析失败: {norm}")
            return False

        # 周末直接返回，避免不必要查询
        if dt.weekday() >= 5:
            self._cache[norm] = False
            # 持久化可选：周末标记很多，不必每次保存；这里仍保存，保证后续快速命中
            self._save_cache()
            return False

        try:
            data_map = xtdata.get_local_data(
                field_list=['time', 'open', 'high', 'low', 'close', 'volume'],
                stock_list=[self.BENCHMARK_TICKER],
                period='1d',
                start_time=norm,
                end_time=norm,
                fill_data=True
            )
            bench_df = data_map.get(self.BENCHMARK_TICKER)
            is_trade = (bench_df is not None) and (not bench_df.empty)
            self._cache[norm] = is_trade
            # 为避免频繁IO，可做简单节流；此处直接保存，确保跨进程可复用
            self._save_cache()
            return is_trade
        except Exception as e:
            # 查询异常时，谨慎返回False并记录
            self._logger.warning(f"查询交易日失败({norm}): {e}")
            self._cache[norm] = False
            self._save_cache()
            return False

    def filter_df_to_trading_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤 DataFrame，仅保留官方交易日索引。

        - 优先使用沪深300在区间内的本地日线索引作为“交易日集合”
        - 若查询失败，退化为移除周末（不处理法定节假日）
        - 保持原始顺序
        """
        try:
            if df is None or df.empty:
                return df
            if not isinstance(df.index, pd.DatetimeIndex):
                # 尽可能兼容：若存在'date'列则设为索引，否则尝试转换现有索引
                if 'date' in df.columns:
                    df = df.set_index('date')
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            start = df.index.min()
            end = df.index.max()
            start_str = start.strftime('%Y%m%d')
            end_str = end.strftime('%Y%m%d')

            # 查询沪深300的本地日线数据，作为“交易日集合”
            data_map = xtdata.get_local_data(
                field_list=['time', 'open', 'high', 'low', 'close', 'volume'],
                stock_list=[self.BENCHMARK_TICKER],
                period='1d',
                start_time=start_str,
                end_time=end_str,
                fill_data=True
            )
            bench_df = data_map.get(self.BENCHMARK_TICKER)
            if bench_df is not None and not bench_df.empty:
                # xtdata索引通常为YYYYMMDD整数/字符串，需要转换为DatetimeIndex
                try:
                    bench_index = pd.to_datetime(bench_df.index.astype(str), format='%Y%m%d')
                except Exception:
                    bench_index = pd.to_datetime(bench_df.index)
                trading_days = pd.Index(bench_index.unique())
                filtered = df.loc[df.index.isin(trading_days)]
                return filtered
            else:
                # 退化：移除周末
                mask = df.index.weekday < 5
                return df.loc[mask]
        except Exception as e:
            self._logger.warning(f"过滤交易日失败: {e}")
            # 发生异常时，保守返回原数据
            return df
