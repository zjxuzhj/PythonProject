import os
from datetime import datetime

import pandas as pd

import getAllStockCsv


def get_stock_data(symbol):
    """带本地缓存的数据获取"""
    file_name = f"stock_{symbol}_20240201.parquet"
    cache_path = os.path.join("data_cache", file_name)

    # 非强制更新时尝试读取缓存
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path, engine='fastparquet')
            print(f"从缓存加载数据：{symbol}")
            return df, True  # 返回缓存标记
        except Exception as e:
            print(f"缓存读取失败：{e}（建议删除损坏文件：{cache_path}）")

    print(f"数据获取失败：{symbol}")
    return pd.DataFrame()


def find_recent_first_limit_up(code, old_df, days=7):
    """识别最近days个交易日内存在的首板涨停日并排除连板"""
    market = "科创板" if code.startswith(("688", "689")) else "创业板" if code.startswith(("300", "301")) else "主板"
    limit_rate = 0.20 if market in ["创业板", "科创板"] else 0.10

    # 获取当前数据的最新日期
    end_date = old_df.index.max()  # 数据的最新交易日
    if end_date is None or old_df.empty:
        return []

    # 筛选有效时间范围
    start_date = (end_date - pd.offsets.BDay(days)).strftime("%Y%m%d")
    date_mask = (old_df.index >= start_date) & (old_df.index <= end_date)
    df = old_df.loc[date_mask].copy()

    # 计算涨停价
    df['prev_close'] = df['close'].shift(1)
    df['prev_close'] = df['prev_close'].fillna(df['close'])
    df['limit_price'] = (df['prev_close'] * (1 + limit_rate)).round(2)

    # 识别所有涨停日（包含过滤条件）
    df['is_limit'] = df['close'] >= df['limit_price']

    # 识别涨停日
    limit_days = df[df['is_limit']].index.tolist()

    # 排除涨停日是最后一天的情况
    last_day = df.index.max()
    limit_days = [day for day in limit_days if day != last_day]
    if not limit_days:
        return []

    # 按时间降序排序（最近的排在前面）
    limit_days.sort(reverse=True)

    valid_days = []
    for day in limit_days:
        # 确认是否为最近的首板，跳过非最近涨停日
        if day != limit_days[0]:
            continue

        # 检查前一日是否涨停（排除连板）
        prev_day = None
        try:
            prev_idx = df.index.get_loc(day) - 1
            if prev_idx >= 0:
                prev_day = df.index[prev_idx]
        except (IndexError, KeyError):
            prev_day = None

        if prev_day and df.loc[prev_day, 'is_limit']:
            # 前一日涨停，排除该涨停日（连板）
            continue

        # 条件1：排除后一日涨停
        next_day = df.index[df.index.get_loc(day) + 1] if (df.index.get_loc(day) + 1) < len(df) else None
        if next_day and df.loc[next_day, 'is_limit']:
            # 后一日涨停，排除该涨停日（连板）
            continue

        # 条件2：涨停后第一天涨幅>8%的排除
        next_day_idx = df.index.get_loc(day) + 1
        if next_day_idx < len(df):
            next_day = df.index[next_day_idx]
            base_price = df.loc[day, 'close']
            if abs(base_price) < 1e-5:
                continue  # 跳过无效数据
            next_day_change = (df.loc[next_day, 'close'] - base_price) / base_price * 100
            # 如果次日涨幅超过8%，排除该首板日
            if next_day_change >= 8:
                continue

        #  条件3：涨停后第一天量能过滤条件（放量存在出货可能）
        if next_day_idx < len(df):
            next_day = df.index[next_day_idx]
            limit_day_volume = df.loc[day, 'volume']
            next_day_volume = df.loc[next_day, 'volume']
            next_day_open = df.loc[next_day, 'open']
            next_day_close = df.loc[next_day, 'close']
            if (next_day_volume >= limit_day_volume * 3.6) and (next_day_close < next_day_open):
                continue

        # 条件4：前五日累计涨幅校验（相当于往前数五根k线，那天的收盘价到涨停当天收盘价的涨幅，也就是除涨停外，四天累计只能涨5%）
        if df.index.get_loc(day) >= 5:
            pre5_start = df.index[df.index.get_loc(day) - 5]
            pre5_close = df.loc[pre5_start, 'close']
            total_change = (df.loc[day, 'close'] - pre5_close) / pre5_close * 100
            if total_change >= 15:
                continue

        # 条件5：前高压制条件
        day_idx = df.index.get_loc(day)
        if day_idx >= 13:  # 确保10日历史数据
            # 计算前高（10日最高价）
            historical_high = df.iloc[day_idx - 10:day_idx]['high'].max()
            # 检查前3日最高价是否触及前高的95%
            recent_3day_high = df.iloc[day_idx - 3:day_idx]['high'].max()
            if historical_high * 0.95 <= recent_3day_high < historical_high:
                continue  # 触发排除条件

        valid_days.append(day)
    return valid_days


def generate_signals(df, first_limit_day, stock_code, stock_name):
    base_price = df.loc[first_limit_day, 'close']  # 首板收盘价，最重要的位置，表示主力的支撑度
    df['5ma'] = df['close'].rolling(5, min_periods=1).mean()
    post_limit_df = df[df.index >= first_limit_day].copy()

    # 涨停后每天收盘价都高于首板收盘价
    always_above_base = True
    for date in post_limit_df.index:
        if date == first_limit_day:
            continue  # 跳过首板日（不需要比较）
        close_price = post_limit_df.loc[date, 'close']
        if close_price < base_price:
            always_above_base = False
            break

    # 判断每日是否触及五日线
    never_touched = True  # 初始化为True

    # 遍历涨停日后的每个交易日
    for date in post_limit_df.index:
        row = post_limit_df.loc[date]
        # 判断是否触及：当日最低价 ≤ 五日线 ≤ 当日最高价
        if row['low'] <= row['5ma'] <= row['high']:
            never_touched = False
            break  # 一旦触及即退出循环

    return never_touched and always_above_base

def get_target_stocks():
    limit_up_stocks = []

    query_tool = getAllStockCsv.StockQuery()
    # 加载股票列表并过滤
    filtered_stocks = query_tool.get_all_filter_stocks()
    stock_list = filtered_stocks[['stock_code', 'stock_name']].values
    total = len(stock_list)

    for idx, (code, name) in enumerate(stock_list, 1):
        df, _ = get_stock_data(code)
        if df.empty:
            continue

        first_limit_days = find_recent_first_limit_up(code, df)
        for day in first_limit_days:
            if generate_signals(df, day, code, name):
                limit_up_stocks.append((code, name, day.strftime("%Y-%m-%d")))

    # 新增分组排序逻辑 ======================
    today = datetime.now().date()
    days_groups = {}

    print(f"\n总计发现 {len(limit_up_stocks)} 只符合要求的股票")

    for stock in limit_up_stocks:
        # 提取日期并转换为日期对象
        code, name, limit_date, = stock
        limit_day = datetime.strptime(limit_date, "%Y-%m-%d").date()
        delta_days = (today - limit_day).days

        # 按天数分组
        if delta_days not in days_groups:
            days_groups[delta_days] = []
        days_groups[delta_days].append(stock)

    # 按天数排序（网页3排序方法）
    sorted_days = sorted(days_groups.items(), key=lambda x: x[0], reverse=False)

    for delta, stocks in sorted_days:
        # 打印数据行
        for stock in stocks:
            code, name, date = stock

            print("  " + "   ".join(stock) + "  ")



if __name__ == '__main__':
    limit_up_stocks = []

    query_tool = getAllStockCsv.StockQuery()
    # 加载股票列表并过滤
    filtered_stocks = query_tool.get_all_filter_stocks()
    stock_list = filtered_stocks[['stock_code', 'stock_name']].values
    total = len(stock_list)

    for idx, (code, name) in enumerate(stock_list, 1):
        df, _ = get_stock_data(code)
        if df.empty:
            continue

        first_limit_days = find_recent_first_limit_up(code, df)
        for day in first_limit_days:
            if generate_signals(df, day, code, name):
                limit_up_stocks.append((code, name, day.strftime("%Y-%m-%d")))

    # 新增分组排序逻辑 ======================
    today = datetime.now().date()
    days_groups = {}

    print(f"\n总计发现 {len(limit_up_stocks)} 只符合要求的股票")

    for stock in limit_up_stocks:
        # 提取日期并转换为日期对象
        code, name, limit_date, = stock
        limit_day = datetime.strptime(limit_date, "%Y-%m-%d").date()
        delta_days = (today - limit_day).days

        # 按天数分组
        if delta_days not in days_groups:
            days_groups[delta_days] = []
        days_groups[delta_days].append(stock)

    # 按天数排序（网页3排序方法）
    sorted_days = sorted(days_groups.items(), key=lambda x: x[0], reverse=False)

    target_stocks = set()

    for delta, stocks in sorted_days:
        # 打印数据行
        for stock in stocks:
            code, name, date = stock
            target_stocks.add(code)
            # print("  " + "   ".join(stock) + "  ")

    print(f"  {list(target_stocks)} ")