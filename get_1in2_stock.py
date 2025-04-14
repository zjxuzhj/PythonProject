import os

import pandas as pd


def is_first_limit_up(symbol, df):
    """综合判断股票当日是否涨停（支持实时与历史数据）"""
    if len(df) < 3:
        print("数据不足，至少需要3个交易日数据")
        return False

    # 获取最新交易日的换手率（新增）
    latest_turnover = df.iloc[-1]['换手率']  # 假设列名为"换手率"，取当日数据

    # 换手率验证（新增核心条件）
    if not (3 <= latest_turnover <= 20):  # 网页2[2](@ref)/网页4[4](@ref)建议的阈值范围
        return False

    # 获取市场类型（网页6的涨跌幅规则）
    market_type = "科创板" if symbol.startswith(("688", "689")) else \
        "创业板" if symbol.startswith(("300", "301")) else "主板"

    # 动态涨跌幅设置（根据2025年最新规则）
    limit_map = {
        "主板": 0.10,
        "创业板": 0.20,
        "科创板": 0.20,
    }
    limit_rate = limit_map[market_type]

    # 计算前一个交易日的涨停价
    prev_close = df.iloc[-3]['close']  # 倒数第3个交易日的收盘价
    limit_up_price_prev = round(prev_close * (1 + limit_rate), 2)

    # 计算当前交易日的涨停价
    current_prev_close = df.iloc[-2]['close']  # 倒数第2个交易日的收盘价
    limit_up_price_current = round(current_prev_close * (1 + limit_rate), 2)

    # 判断前一个交易日是否涨停
    prev_day_close = df.iloc[-2]['close']  # 前一个交易日的收盘价
    is_prev_day_limit_up = prev_day_close >= limit_up_price_prev

    # 判断当前交易日是否涨停
    latest = df.iloc[-1]  # 当前交易日的数据
    is_current_day_limit_up = latest['close'] >= limit_up_price_current

    # 计算近3日累计涨幅（从倒数第4个交易日到当前交易日）
    start_close = df.iloc[-4]['close']  # 倒数第4个交易日的收盘价
    end_close = latest['close']  # 当前交易日的收盘价
    cumulative_return = (end_close - start_close) / start_close

    # 首板判断条件
    # return is_current_day_limit_up and not is_prev_day_limit_up

    # 判断条件：
    # 1. 当前涨停且前一日未涨停（首板涨停条件）
    # 2. 近3日累计涨幅≤25%
    return (
            is_current_day_limit_up
            and not is_prev_day_limit_up
            and (cumulative_return <= 0.25)
            and (3 <= latest_turnover <= 20)  # 网页5[5](@ref)对换手率区间的验证要求
    )


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


def filter_stocks(df):
    df['clean_code'] = df['stock_code'].str.extract(r'(\d{6})')[0]  # 提取纯数字代码
    is_bse = df['clean_code'].str.startswith(('43', '83', '87', '88', '92','30','68'))
    is_st = df['stock_name'].str.contains(r'ST|\*ST|退市', na=False)
    return df[~is_bse & ~is_st]


if __name__ == '__main__':
    # 参数设置
    symbol = 'sh601086'  # 平安银行
    start_date = '20240201'

    # 初始化涨停股容器
    limit_up_stocks = []
    # 加载股票列表并过滤
    all_stocks = pd.read_csv('stock_code_name.csv')
    filtered_stocks = filter_stocks(all_stocks)
    stock_list = filtered_stocks[['stock_code', 'stock_name']].values
    total = len(stock_list)

    for idx, (code, name) in enumerate(stock_list, 1):
        try:
            # 获取含今日的最新行情（网页1、网页3）
            df, _ = get_stock_data(code, start_date=start_date)  # 获取近两日数据

            # 执行涨停判断（网页2）
            if is_first_limit_up(code, df):
                limit_up_stocks.append((code, name))
                print(f"\033[32m涨停发现：{name}({code})\033[0m")

            # 进度提示（每50只提示）
            if idx % 50 == 0:
                print(f"已扫描{idx}/{len(stock_list)}只，当前涨停数：{len(limit_up_stocks)}")

        except Exception as e:
            print(f"处理异常：{name}({code}) - {str(e)}")
            continue

        # 结果输出（网页5）
    print("\n\033[1m===== 今日涨停统计 =====\033[0m")
    print(f"涨停总数：\033[31m{len(limit_up_stocks)}\033[0m只")
    for code, name in limit_up_stocks:
        print(f"· {name}({code})")
