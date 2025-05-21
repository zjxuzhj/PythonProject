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


def modify_last_days_and_calc_ma5(df, two_days_mode=False):
    """参数说明：
    two_days_mode : bool (默认False)
        是否启用双日模式（修改最后两日数据）
    """
    # 数据校验（参考网页6的参数校验思想[6](@ref)）
    if df.empty or len(df) < (2 if two_days_mode else 1):
        raise ValueError(f"数据不足，至少需要{2 if two_days_mode else 1}个交易日数据")

    modified_df = df.copy().sort_index(ascending=True)

    # 🌟 动态修改天数（网页8的可变参数思想[8](@ref)）
    days_to_modify = 2 if two_days_mode else 1

    # 倒序修改最后N日数据（网页4的滚动计算思想[4](@ref)）
    for i in range(1, days_to_modify + 1):
        close_col = modified_df.columns.get_loc('close')
        modified_df.iloc[-i, close_col] = modified_df['close'].iloc[-i] * 1.03

    # 计算MA5（网页2的rolling方法[2](@ref)）
    modified_df['MA5'] = modified_df['close'].rolling(
        window=5,
        min_periods=1
    ).mean().round(2)

    return modified_df

if __name__ == '__main__':
    today = datetime.now()
    start_date = '20240201'

    query_tool = getAllStockCsv.StockQuery()
    filtered_stocks = query_tool.get_all_filter_stocks()
    stock_list = filtered_stocks[['stock_code', 'stock_name']].values

    target_stocks = {'sz000605', 'sh600397', 'sz002730'}
    two_days_mode=True

    all_signals = []
    for idx, (code, name) in enumerate(stock_list, 1):
        if code not in target_stocks:
            continue  # 跳过非目标股票

        df, _ = get_stock_data(code, start_date=start_date)
        if df.empty:
            continue

        modified_df = modify_last_days_and_calc_ma5(df, two_days_mode=two_days_mode)

        # 获取最新MA5值
        latest_data = modified_df.iloc[- (2 if two_days_mode else 1):]

        # 记录信号
        all_signals.append({
            '代码': code,
            '名称': name,
            '最新收盘': latest_data['close'].values[-1],
            'MA5': modified_df['MA5'].iloc[-1],
            # 🌟 新增双日模式特有字段（网页1的条件判断[1](@ref)）
            **({'前日收盘': latest_data['close'].values[-2]} if two_days_mode else {})
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
