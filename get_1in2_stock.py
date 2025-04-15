import os
import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime
import re
import getAllStockCsv

def calculate_premium_rate(symbol, df, days=100, method='open'):
    """计算指定天数内的平均涨停次日溢价率（支持多种计算方式）"""
    # 参数校验
    if len(df) < 2:
        return 0.0  # 数据不足时返回默认值

    # 获取市场类型
    market_type = "科创板" if symbol.startswith(("688", "689")) else \
        "创业板" if symbol.startswith(("300", "301")) else "主板"
    limit_rate = 0.20 if market_type in ["创业板", "科创板"] else 0.10

    # 遍历历史数据识别涨停日[1](@ref)
    limit_up_dates = []
    for i in range(2, len(df)):
        prev_close = df.iloc[i - 2]['close']
        current_close = df.iloc[i - 1]['close']
        # 涨停价计算（精确到分）[7](@ref)
        limit_price = round(prev_close * (1 + limit_rate), 2)
        if current_close >= limit_price:
            limit_up_dates.append(df.index[i - 1])

    # 计算每个涨停次日的溢价率
    premiums = []
    premium = 0.0
    for date in limit_up_dates[-days:]:  # 取最近days天
        # 查找次日数据[2](@ref)
        next_day = df[df.index > date].iloc[0] if not df[df.index > date].empty else None
        if next_day is not None:
            limit_price = round(df[df.index == date]['close'].values[0], 2)
            # 选择计算方式[3](@ref)
            if method == 'open':
                premium = (next_day['open'] - limit_price) / limit_price * 100
            elif method == 'close':
                premium = (next_day['close'] - limit_price) / limit_price * 100
            premiums.append(premium)

    return round(np.mean(premiums), 2) if premiums else 0.0

def is_first_limit_up(symbol, df, query_tool):
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

    market_value = query_tool.get_stock_market_value(query_tool.get_simple_by_code(symbol))

    # 判断条件：
    # 1. 当前涨停且前一日未涨停（首板涨停条件）
    # 2. 近3日累计涨幅≤25%
    # 3. 首板3%≤换手率≤20%
    # 4. 股票10<市值<150
    return (
            is_current_day_limit_up
            and not is_prev_day_limit_up
            and (cumulative_return <= 0.25)
            and (3 <= latest_turnover <= 20)
            and (10 <= market_value <= 150)
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


def format_limit_time(time_str):
    """格式化六位数字为HH:MM:SS"""
    try:
        # 基础校验：6位纯数字
        if len(time_str) != 6 or not time_str.isdigit():
            return "09:25:00"  # 默认值

        # 提取时分秒
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])

        # 合法性校验
        if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
            return f"{hour:02d}:{minute:02d}:{second:02d}"
        else:
            return "09:25:00"  # 非法时间返回默认值
    except:
        return "09:25:00"

def filter_stocks(df):
    df['clean_code'] = df['stock_code'].str.extract(r'(\d{6})')[0]  # 提取纯数字代码
    is_bse = df['clean_code'].str.startswith(('43', '83', '87', '88', '92','30','68'))
    is_st = df['stock_name'].str.contains(r'ST|\*ST|退市', na=False)
    return df[~is_bse & ~is_st]


if __name__ == '__main__':
    # 获取当日涨停数据（新增）
    today = datetime.now().strftime("%Y%m%d")
    zt_df = ak.stock_zt_pool_em(date=today)

    # 创建代码映射字典（关键优化）
    zt_time_map = dict(zip(
        zt_df['代码'].astype(str),  # 确保代码为字符串类型
        zt_df['首次封板时间']
    ))

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
    query_tool = getAllStockCsv.StockQuery()

    for idx, (code, name) in enumerate(stock_list, 1):
        # try:
            # 获取含今日的最新行情（网页1、网页3）
            df, _ = get_stock_data(code, start_date=start_date)  # 获取近两日数据

            # 执行涨停判断（网页2）
            if is_first_limit_up(code, df, query_tool):
                premium_rate = calculate_premium_rate(code, df, days=100, method='open')
                limit_up_stocks.append((code, name,premium_rate))
                print(f"\033[32m涨停发现：{name}({code})\033[0m")

            # 进度提示（每50只提示）
            if idx % 50 == 0:
                print(f"已扫描{idx}/{len(stock_list)}只，当前涨停数：{len(limit_up_stocks)}")

        # except Exception as e:
        #     print(f"处理异常：{name}({code}) - {str(e)}")
        #     continue

        # 结果输出（网页5）
    print("\n\033[1m===== 今日涨停统计 =====\033[0m")
    print(f"涨停总数：\033[31m{len(limit_up_stocks)}\033[0m只")
    # 新增排序逻辑
    sorted_stocks = sorted(limit_up_stocks, key=lambda x: x[2], reverse=True)
    for code, name, premium_rate in sorted_stocks:
        clean_code = re.sub(r'\D', '', code)  # 移除非数字字符
        first_time = zt_time_map.get(clean_code, '09:25:00')  # 默认值处理
        print(f"· {name}({code}) ｜ 百日溢价率：{premium_rate}% ｜ 涨停时间：{format_limit_time(first_time)}")
