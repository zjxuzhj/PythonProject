# utils/data_utils.py
import pandas as pd
import os
from datetime import datetime, date
import tempfile
from typing import Iterable, Optional, Union
import getAllStockCsv as tools


class StockDataCacheCleaner:
    def __init__(self, cache_dir: str = "data_cache", base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.cache_dir = cache_dir

    def delete_date_data(
        self,
        target_date: Union[str, datetime, date],
        symbols: Optional[Iterable[str]] = None,
        file_date_suffix: Optional[str] = None,
    ) -> dict:
        target_ts = self._to_normalized_timestamp(target_date)
        cache_path = os.path.join(self.base_dir, self.cache_dir)
        if not os.path.isdir(cache_path):
            return {"deleted_rows": 0, "touched_files": 0, "cache_path": cache_path}

        file_paths = self._collect_cache_files(cache_path, symbols=symbols, file_date_suffix=file_date_suffix)

        deleted_rows = 0
        touched_files = 0
        for file_path in file_paths:
            removed = self._delete_date_from_parquet(file_path, target_ts)
            if removed > 0:
                deleted_rows += removed
                touched_files += 1

        return {"deleted_rows": deleted_rows, "touched_files": touched_files, "cache_path": cache_path}

    def delete_date_data_for_symbol(
        self,
        symbol: str,
        target_date: Union[str, datetime, date],
        file_date_suffix: Optional[str] = None,
    ) -> dict:
        return self.delete_date_data(target_date, symbols=[symbol], file_date_suffix=file_date_suffix)

    @staticmethod
    def _to_normalized_timestamp(target_date: Union[str, datetime, date]) -> pd.Timestamp:
        if isinstance(target_date, (datetime, date)):
            return pd.Timestamp(target_date).normalize()

        s = str(target_date).strip()
        if len(s) == 8 and s.isdigit():
            return pd.Timestamp(datetime.strptime(s, "%Y%m%d")).normalize()

        if len(s) == 10 and (s[4] in "-/" and s[7] in "-/"):
            s = s.replace("/", "-")
            return pd.Timestamp(datetime.strptime(s, "%Y-%m-%d")).normalize()

        return pd.to_datetime(s).normalize()

    @staticmethod
    def _collect_cache_files(
        cache_path: str,
        symbols: Optional[Iterable[str]],
        file_date_suffix: Optional[str],
    ) -> list[str]:
        suffix_part = f"_{file_date_suffix}" if file_date_suffix else "_*"

        if symbols:
            file_paths: list[str] = []
            for sym in symbols:
                pattern = f"stock_{sym}{suffix_part}.parquet"
                for name in os.listdir(cache_path):
                    if name == pattern or (file_date_suffix is None and name.startswith(f"stock_{sym}_") and name.endswith(".parquet")):
                        file_paths.append(os.path.join(cache_path, name))
            return sorted(set(file_paths))

        return sorted(
            os.path.join(cache_path, name)
            for name in os.listdir(cache_path)
            if name.startswith("stock_") and name.endswith(".parquet") and (file_date_suffix is None or name.endswith(f"_{file_date_suffix}.parquet"))
        )

    def _delete_date_from_parquet(self, file_path: str, target_ts: pd.Timestamp) -> int:
        try:
            df = pd.read_parquet(file_path, engine="fastparquet")
        except Exception:
            return 0

        new_df, removed = self._drop_date_from_df(df, target_ts)
        if removed <= 0:
            return 0

        parent_dir = os.path.dirname(file_path)
        fd, tmp_path = tempfile.mkstemp(prefix="tmp_", suffix=".parquet", dir=parent_dir)
        os.close(fd)
        try:
            new_df.to_parquet(tmp_path, engine="fastparquet")
            os.replace(tmp_path, file_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return 0

        return removed

    @staticmethod
    def _drop_date_from_df(df: pd.DataFrame, target_ts: pd.Timestamp) -> tuple[pd.DataFrame, int]:
        if df.empty:
            return df, 0

        if "date" in df.columns:
            date_col = pd.to_datetime(df["date"], errors="coerce")
            mask = date_col.dt.normalize() == target_ts
            removed = int(mask.sum())
            return df.loc[~mask].copy(), removed

        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            converted = pd.to_datetime(idx, errors="coerce")
            df = df.copy()
            df.index = converted

        if not isinstance(df.index, pd.DatetimeIndex):
            return df, 0

        mask = df.index.normalize() == target_ts
        removed = int(mask.sum())
        return df.loc[~mask].copy(), removed


# 获得今天以前的数据
def get_stock_data_not_today(symbol, isNeedLog):
    file_name = f"stock_{symbol}_20240201.parquet"
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cache_path = os.path.join(root_dir, "data_cache", file_name)

    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path, engine='fastparquet')

            # <--- 新增逻辑：确保索引是日期时间格式 ---
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # <--- 新增逻辑：过滤掉今天及之后的数据 ---
            today_timestamp = pd.Timestamp.now().normalize()
            yesterday_timestamp = today_timestamp - pd.Timedelta(days=1)
            day_before_yesterday_timestamp = today_timestamp - pd.Timedelta(days=2)
            # 2. 直接与索引进行比较，筛选出今天之前的所有数据
            df = df[df.index < today_timestamp]

            if isNeedLog:
                print(f"从缓存加载数据 (已过滤今日)：{symbol}")
            return df, True
        except Exception as e:
            print(f"缓存读取或过滤失败：{e}（建议删除损坏文件：{cache_path}）")

    print(f"数据获取失败：{symbol}")
    return pd.DataFrame(), False  # <--- 修正：当返回空DF时，成功标记应为False

def get_stock_data(symbol, isNeedLog):
    file_name = f"stock_{symbol}_20240201.parquet"
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cache_path = os.path.join(root_dir, "data_cache", file_name)

    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path, engine='fastparquet')
            if isNeedLog:
                print(f"从缓存加载数据：{symbol}")
            return df, True
        except Exception as e:
            print(f"缓存读取失败：{e}（建议删除损坏文件：{cache_path}）")

    print(f"数据获取失败：{symbol}")
    return pd.DataFrame(), False

def get_ma5_price(stock_code, current_date=None, current_price=None):
    """
        获取指定股票的最新MA5价格，支持动态更新当日数据
        current_date (datetime, optional): 当前日期（用于动态更新）
        current_price (float, optional): 当前价格（用于动态计算）
    """
    pure_code = tools.convert_stock_code(stock_code)
    df, _ = get_stock_data(pure_code, False)

    if df.empty or len(df) < 4:
        print(f"警告：{stock_code} 数据不足，无法计算MA5")
        return None

    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if current_date and current_price:
            current_date = pd.Timestamp(current_date)
            # 情况1：已有当日数据 → 更新收盘价
            if current_date in df.index:
                df.loc[df.index == current_date, 'close'] = current_price
            # 情况2：无当日数据 → 添加新行
            else:
                new_row = df.iloc[-1].copy()
                new_row.name = current_date
                new_row['close'] = current_price
                df = pd.concat([df, new_row.to_frame().T])
        df = df.sort_index(ascending=True)
        df['MA5'] = df['close'].rolling(
            window=5,
            min_periods=5
        ).mean().round(2)
        return df['MA5'].iloc[-1]
    except KeyError as e:
        print(f"数据列缺失错误：{str(e)}")
        return None
    except Exception as e:
        print(f"MA5计算异常：{str(e)}")
        return None

def modify_last_days_and_calc_ma5(df, predict_ratio=1.04):
    """模拟预测MA5的核心方法"""
    if df.empty or len(df) < 2:
        raise ValueError("数据不足，至少需要2个交易日数据")

    modified_df = df.copy().sort_index(ascending=True)
    modified_df['adjusted_close'] = modified_df['close']

    # 复制最后一行并应用预测涨幅（原1.04倍逻辑）
    new_row = modified_df.iloc[-1].copy()
    new_row['close'] *= predict_ratio  # 动态传入预测系数
    new_row.name = new_row.name + pd.Timedelta(days=1)
    modified_df = pd.concat([modified_df, new_row.to_frame().T], axis=0)

    # 计算修正后的MA5
    modified_df['MA5'] = modified_df['close'].rolling(
        window=5, min_periods=1
    ).mean().round(2)
    return modified_df



