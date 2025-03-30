import os
import time

import akshare as ak
import pandas as pd
from fastparquet import write
from fastparquet import ParquetFile

def updateStockData(symbol, start_date):
    file_name = f"stock_{symbol}_{start_date}.parquet"
    cache_path = os.path.join("data_cache", file_name)

def addStockData(symbol, start_date):
    file_name = f"stock_{symbol}_{start_date}.parquet"
    cache_path = os.path.join("data_cache", file_name)
    df = pd.read_parquet(cache_path).reset_index()
    # 新增数据构建（网页2的DataFrame创建方法）
    new_row = {
        "date": pd.to_datetime("2025-3-29"),
        "股票代码": "600563",
        "open": 98.5,
        "close": 98.8,
        "high": 99.3,
        "low": 97.9,
        "volume": 19500,
        "成交额": 192000000,
        "振幅": 2.85,
        "涨跌幅": 1.24,
        "涨跌额": 1.2,
        "换手率": 0.91
    }
    # 追加数据
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    # 保存（带日期索引）
    df.set_index('date').to_parquet(
        cache_path,
        engine='fastparquet',
        compression='snappy'
    )




# 使用示例（单独执行此类时）
if __name__ == "__main__":
    addStockData('600563','20240501')

    file_name = f"stock_600563_20240501.parquet"
    cache_path = os.path.join("data_cache", file_name)
    existing_data = pd.read_parquet(cache_path, engine='fastparquet')
    print("111")

