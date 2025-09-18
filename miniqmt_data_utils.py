# utils/data_utils.py
import pandas as pd
import os
from datetime import datetime
import getAllStockCsv as tools


# 获得今天以前的数据
def get_stock_data_not_today(symbol, isNeedLog):
    file_name = f"stock_{symbol}_20240201.parquet"
    cache_path = os.path.join("data_cache", file_name)

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
    cache_path = os.path.join("data_cache", file_name)

    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path, engine='fastparquet')
            if isNeedLog:
                print(f"从缓存加载数据：{symbol}")
            return df, True
        except Exception as e:
            print(f"缓存读取失败：{e}（建议删除损坏文件：{cache_path}）")

    print(f"数据获取失败：{symbol}")
    return pd.DataFrame()

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


