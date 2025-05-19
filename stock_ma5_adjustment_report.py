import os
from datetime import datetime

import pandas as pd

import getAllStockCsv


def get_stock_data(symbol, start_date, force_update=False):
    """带本地缓存的数据获取"""
    # 生成唯一文件名（网页1）
    file_name = f"stock_{symbol}_{start_date}.parquet"
    cache_path = os.path.join("data_cache", file_name)

    # 非强制更新时尝试读取缓存
    if not force_update and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path, engine='fastparquet')
            print(f"从缓存加载数据：{symbol}")
            return df, True  # 返回缓存标记
        except Exception as e:
            print(f"缓存读取失败：{e}（建议删除损坏文件：{cache_path}）")

    # 强制更新或缓存不存在时获取新数据（网页7）
    print(f"数据获取失败：{symbol}")
    return pd.DataFrame()


def modify_last_day_and_calc_ma5(df):
    """修改最后一日为3%涨幅并计算五日线"""
    if df.empty or len(df) < 1:
        return df
    # 深拷贝避免污染原始数据
    modified_df = df.copy()
    # 按日期排序确保最后一行是最新数据（网页7关键步骤）
    modified_df.sort_index(ascending=True, inplace=True)
    # 获取最后交易日的收盘价（网页6数据操作）
    last_close = modified_df['close'].iloc[-1]
    # 计算3%涨幅后的新收盘价（网页2数值计算）
    new_close = round(last_close * 1.03, 2)
    # 修改最后一行数据（网页1的replace高级用法）
    modified_df.iloc[-1, modified_df.columns.get_loc('close')] = new_close
    # 计算五日移动平均线（网页6、7、8核心方法）
    modified_df['MA5'] = modified_df['close'].rolling(window=5, min_periods=1).mean()
    return modified_df


if __name__ == '__main__':
    today = datetime.now()
    start_date = '20240201'

    query_tool = getAllStockCsv.StockQuery()
    filtered_stocks = query_tool.get_all_filter_stocks()
    stock_list = filtered_stocks[['stock_code', 'stock_name']].values

    target_stocks = {'sh605198', 'sh600448', 'sz002730'}

    all_signals = []
    for idx, (code, name) in enumerate(stock_list, 1):
        if code not in target_stocks:
            continue  # 跳过非目标股票

        df, _ = get_stock_data(code, start_date=start_date)
        if df.empty:
            continue

        modified_df = modify_last_day_and_calc_ma5(df)

        # 获取最新MA5值
        latest_ma5 = modified_df['MA5'].iloc[-1]

        # 记录信号
        all_signals.append({
            '股票代码': code,
            '股票名称': name,
            '收盘价': modified_df['close'].iloc[-1],
            '五日线': round(latest_ma5, 2),
        })

    if len(all_signals) > 0:
        result_df = pd.DataFrame(all_signals)
        print("\n目标股票五日线报告：")
        print(result_df.to_string(index=False))
    else:
        print("没有找到目标股票的有效数据，请检查：")
        print("1. 股票代码是否包含sh/sz前缀（网页3的编码规范）")
        print("2. 缓存文件是否存在完整收盘价数据（网页6的parquet格式要求）")
        print("3. 股票代码是否有效（如sh605198是华新水泥的B股代码）")
