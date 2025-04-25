import os
import re
from datetime import datetime, timedelta

import akshare as ak
import numpy as np
import pandas as pd

import getAllStockCsv


def check_recent_limit_up(symbol, df, days=20, isBackTest=False):
    day = 1
    if isBackTest: day = 2
    """检查最近N个交易日内是否有涨停"""
    if len(df) < days:
        return 0  # 数据不足时返回0

    # 截取最近days个交易日的数据
    recent_df = df.iloc[-(days+ day):-day]

    # 获取市场类型
    market_type = "科创板" if symbol.startswith(("688", "689")) else \
        "创业板" if symbol.startswith(("300", "301")) else "主板"
    limit_rate = 0.20 if market_type in ["创业板", "科创板"] else 0.10

    # 遍历判断每个交易日
    limit_count = 0
    for i in range(1, len(recent_df)):
        prev_close = recent_df.iloc[i - 1]['close']
        current_close = recent_df.iloc[i]['close']
        limit_price = round(prev_close * (1 + limit_rate), 2)
        if current_close >= limit_price:
            limit_count += 1

    return limit_count

def calculate_today_change(df):
    """计算今日涨跌幅（与前收盘价对比）"""
    if len(df) < 2:  # 至少需要昨日和今日数据
        return 0.0, 0.0

    # 获取昨日收盘价和今日收盘价[1,2](@ref)
    prev_close = df.iloc[-2]['close']  # 昨日收盘价
    today_close = df.iloc[-1]['close']  # 今日收盘价

    # 计算涨跌幅[4,7](@ref)
    change = (today_close - prev_close) / prev_close * 100
    return round(change, 2)


def calculate_auction_return(df):
    """计算竞价买入到收盘的收益率"""
    if len(df) < 1:  # 至少需要今日数据
        return 0.0

    # 获取今日开盘价和收盘价[3,8](@ref)
    today_open = df.iloc[-1]['open']  # 今日开盘价（竞价买入价）
    today_close = df.iloc[-1]['close']  # 今日收盘价

    # 计算收益率[5](@ref)
    return_rate = (today_close - today_open) / today_open * 100
    return round(return_rate, 2)


def calculate_continuation_rate(symbol, df, days=100):
    """计算百日连板率：涨停次日继续涨停的概率"""
    if len(df) < 3:
        return 0.0

    # 识别所有涨停日
    market_type = "科创板" if symbol.startswith(("688", "689")) else \
        "创业板" if symbol.startswith(("300", "301")) else "主板"
    limit_rate = 0.20 if market_type in ["创业板", "科创板"] else 0.10

    # 向量化计算涨停日（优化性能）
    df['prev_close'] = df['close'].shift(2)
    df['limit_price'] = (df['prev_close'] * (1 + limit_rate)).round(2)
    limit_up_days = df[df['close'] >= df['limit_price']].index.tolist()

    # 统计有效涨停日（过滤新股）
    valid_days = [d for d in limit_up_days[-days:] if pd.notnull(df.loc[d]['prev_close'])]

    # 计算连板次数
    continuation_count = 0
    for date in valid_days:
        next_day = df[df.index > date]
        if len(next_day) > 0:
            next_close = next_day.iloc[0]['close']
            next_limit_price = round(df.loc[date]['close'] * (1 + limit_rate), 2)
            if next_close >= next_limit_price:
                continuation_count += 1

    return round(continuation_count / len(valid_days) * 100, 2) if valid_days else 0.0


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


def is_first_limit_up(symbol, df, query_tool, isBackTest=False):
    day = 0
    if isBackTest: day = 1
    """综合判断股票当日是否涨停（支持实时与历史数据）"""
    if len(df) < 3:
        print("数据不足，至少需要3个交易日数据")
        return False

    # 获取最新交易日的换手率（新增）
    latest_turnover = df.iloc[-(1 + day)]['换手率']  # 假设列名为"换手率"，取当日数据

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
    prev_close = df.iloc[-(3 + day)]['close']  # 倒数第3个交易日的收盘价
    limit_up_price_prev = round(prev_close * (1 + limit_rate), 2)

    # 计算当前交易日的涨停价
    current_prev_close = df.iloc[-(2 + day)]['close']  # 倒数第2个交易日的收盘价
    limit_up_price_current = round(current_prev_close * (1 + limit_rate), 2)

    # 判断前一个交易日是否涨停
    prev_day_close = df.iloc[-(2 + day)]['close']  # 前一个交易日的收盘价
    is_prev_day_limit_up = prev_day_close >= limit_up_price_prev

    # 判断当前交易日是否涨停
    latest = df.iloc[-(1 + day)]  # 当前交易日的数据
    is_current_day_limit_up = latest['close'] >= limit_up_price_current

    # 计算近3日累计涨幅（从倒数第4个交易日到当前交易日）
    start_close = df.iloc[-(4 + day)]['close']  # 倒数第4个交易日的收盘价
    end_close = latest['close']  # 当前交易日的收盘价
    cumulative_return = (end_close - start_close) / start_close

    # 首板判断条件
    # return is_current_day_limit_up and not is_prev_day_limit_up

    market_value = query_tool.get_stock_market_value(symbol)

    # 判断条件：
    # 1. 当前涨停且前一日未涨停（首板涨停条件）
    # 2. 近3日累计涨幅≤25%（强势股会超过，去掉）
    # 3. 首板3%≤换手率≤20%
    # 4. 股票10<市值<150
    return (
            is_current_day_limit_up
            and not is_prev_day_limit_up
            # and (cumulative_return <= 0.25)
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

isBackTest = False

if __name__ == '__main__':
    # 获取当日涨停数据（新增）
    today = datetime.now()
    today_str = today.strftime("%Y%m%d")
    yesterday = today - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y%m%d")
    # yesterday_str = yesterday.strftime("20250418")

    if isBackTest:
        zt_df = ak.stock_zt_pool_em(date=yesterday_str)
    else:
        zt_df = ak.stock_zt_pool_em(date=yesterday_str)

    # 创建代码映射字典（关键优化）
    zt_time_map = dict(zip(zt_df['代码'].astype(str), zt_df['首次封板时间']))
    zt_zb_map = dict(zip(zt_df['代码'].astype(str), zt_df['炸板次数']))  # 新增炸板次数映射

    # 参数设置
    symbol = 'sh601086'  # 平安银行
    start_date = '20240201'

    # 初始化涨停股容器
    limit_up_stocks = []

    query_tool = getAllStockCsv.StockQuery()
    # 加载股票列表并过滤
    filtered_stocks = query_tool.get_all_filter_stocks()
    stock_list = filtered_stocks[['stock_code', 'stock_name']].values
    total = len(stock_list)


    for idx, (code, name) in enumerate(stock_list, 1):
        # try:
        # 获取含今日的最新行情（网页1、网页3）
        df, _ = get_stock_data(code, start_date=start_date)  # 获取近两日数据

        # 执行涨停判断（网页2）
        if is_first_limit_up(code, df, query_tool, isBackTest):
            premium_rate = calculate_premium_rate(code, df, days=100, method='open')
            continuation_rate = calculate_continuation_rate(code, df)
            today_change = calculate_today_change(df)
            auction_ret = calculate_auction_return(df)
            # 新增20日涨停次数统计
            recent_limit = check_recent_limit_up(code, df, 10, isBackTest)
            limit_up_stocks.append((code, name, premium_rate, continuation_rate, today_change, auction_ret, recent_limit))
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
    # 按涨停时间排序
    sorted_stocks = sorted(limit_up_stocks, key=lambda x: zt_time_map.get(re.sub(r'\D', '', x[0]), '09:25:00'))

    # 按百日溢价率降序排序
    # sorted_stocks = sorted(limit_up_stocks, key=lambda x: x[2], reverse=True)  # x[2]对应premium_rate

    # 按百日连板率降序排序
    # sorted_stocks = sorted(limit_up_stocks, key=lambda x: x[3], reverse=True)  # x[3]对应continuation_rate

    for code, name, premium_rate, continuation_rate, today_change, auction_ret, recent_limit in sorted_stocks:
        clean_code = re.sub(r'\D', '', code)  # 移除非数字字符
        first_time = zt_time_map.get(clean_code, '09:25:00')  # 默认值处理
        zb_count = zt_zb_map.get(clean_code, 0)  # 获取炸板次数

        # 新增颜色控制逻辑[1,5](@ref)
        RED = '\033[31m'
        GREEN = '\033[32m'
        RESET = '\033[0m'

        # 处理今日涨幅颜色
        today_color = RED if today_change > 0 else GREEN if today_change < 0 else ''
        today_display = f"{today_color}今日涨幅：{today_change}%{RESET if today_color else ''}"

        # 处理竞价收益颜色
        auction_color = RED if auction_ret > 0 else GREEN if auction_ret < 0 else ''
        auction_display = f"{auction_color}竞价收益：{auction_ret}%{RESET if auction_color else ''}"

        output_parts = [f"· {name}({code})", f"百日溢价率：{premium_rate}%", f"百日连板率：{continuation_rate}%",
                        f"涨停时间：{format_limit_time(first_time)}"]
        if isBackTest:
            output_parts.append(today_display)
            output_parts.append(auction_display)

        if zb_count > 0:
            output_parts.append(f"炸板次数：{zb_count}次")

        if recent_limit > 0:
            output_parts.append(f"10日涨停：{recent_limit}次")

        print(" ｜ ".join(output_parts))
