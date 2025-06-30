import os
from datetime import datetime, time

import pandas as pd

import getAllStockCsv


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


def find_recent_first_limit_up(code, old_df, days=7):
    """识别最近days个交易日内存在的首板涨停日并排除连板"""
    market = "科创板" if code.startswith(("688", "689")) else "创业板" if code.startswith(("300", "301")) else "主板"
    limit_rate = 0.20 if market in ["创业板", "科创板"] else 0.10

    # 获取当前数据的最新日期
    end_date = old_df.index.max()
    if end_date is None or old_df.empty:
        return []

    # 筛选有效时间范围
    extended_days = 20
    start_date = (end_date - pd.offsets.BDay(extended_days)).strftime("%Y%m%d")
    date_mask = (old_df.index >= start_date) & (old_df.index <= end_date)
    df = old_df.loc[date_mask].copy()

    # 计算涨停价
    df['prev_close'] = df['close'].shift(1)
    df['prev_close'] = df['prev_close'].fillna(df['close'])
    df['limit_price'] = (df['prev_close'] * (1 + limit_rate)).round(2)

    # 识别所有涨停日（包含过滤条件）
    df['is_limit'] = df['close'] >= df['limit_price']

    # 筛选最近days个交易日内的涨停日（核心筛选范围）
    recent_days_mask = (df.index > (end_date - pd.offsets.BDay(days)).strftime("%Y%m%d")) & (df.index <= end_date)
    limit_days = df[df['is_limit'] & recent_days_mask].index.tolist()

    # 排除涨停日是最后一天的情况
    last_day = df.index.max()
    last_day_is_limit = df.loc[last_day, 'is_limit']
    limit_days = [day for day in limit_days if day != last_day]
    if not limit_days or last_day_is_limit:
        return []

    # 按时间降序排序（最近的排在前面）
    limit_days.sort(reverse=True)

    valid_days = []
    for day in limit_days:
        # 确认是否为最近的首板，跳过非最近涨停日
        if day != limit_days[0]:
            continue

        # 条件0：排除前一日涨停
        prev_day = None
        try:
            prev_idx = df.index.get_loc(day) - 1
            if prev_idx >= 0:
                prev_day = df.index[prev_idx]
        except (IndexError, KeyError):
            prev_day = None

        if prev_day and df.loc[prev_day, 'is_limit']:
            continue

        # 条件1：排除后一日涨停
        next_day = df.index[df.index.get_loc(day) + 1] if (df.index.get_loc(day) + 1) < len(df) else None
        if next_day and df.loc[next_day, 'is_limit']:
            continue

        # 条件2：涨停后第一天涨幅>8%的排除
        next_day_idx = df.index.get_loc(day) + 1
        if next_day_idx < len(df):
            next_day = df.index[next_day_idx]
            base_price = df.loc[day, 'close']
            if abs(base_price) < 1e-5:
                continue  # 跳过无效数据
            next_day_change = (df.loc[next_day, 'close'] - base_price) / base_price * 100
            if next_day_change >= 8:
                continue

        # 条件3：涨停后第一天量能过滤条件（放量存在出货可能）
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
            recent_3day_high = df.iloc[-3:]['high'].max()  # 最后3天
            if historical_high * 0.95 <= recent_3day_high < historical_high:
                continue

        # 条件6：排除首板后第一个交易日放量阳线+第二个交易日低开未收复前日实体中点的情况
        if next_day_idx + 1 < len(df):  # 确保有首板第二个交易日数据
            # 获取首板次日（第一天）和次次日（第二天）数据
            first_day = df.index[next_day_idx]
            first_day_data = df.loc[first_day]
            second_day = df.index[next_day_idx + 1]
            second_day_data = df.loc[second_day]

            # 条件6-1：首板次日为放量实体阳线（成交量>首板日且实体占比在总的价格范围的>50%）
            volume_condition = (first_day_data['volume'] > df.loc[day, 'volume'] * 1.5)  # 放量1.5倍
            price_range = first_day_data['high'] - first_day_data['low']
            if abs(price_range) < 1e-5:  # 若最高价=最低价（一字线），实体占比无法计算，直接排除
                candle_condition = False
            else:
                # 计算实体部分（收盘价 - 开盘价）
                candle_body = abs(first_day_data['close'] - first_day_data['open'])
                # 计算实体占总价格范围的比例
                body_ratio = candle_body / price_range
                # 阳线且实体占比超过50%
                candle_condition = (first_day_data['close'] > first_day_data['open']) and (body_ratio > 0.5)

            # 条件6-2：第二日低开且未收复前日实体中点
            # 计算实体中点：开盘价和收盘价的平均（因为是阳线）
            midpoint = (first_day_data['open'] + first_day_data['close']) / 2  # 前日阳线实体中点
            # 第二日低开：开盘低于前一日收盘
            low_open_condition = (second_day_data['open'] < first_day_data['close'])
            # 未收复实体中点：收盘价低于实体中点
            recover_condition = (second_day_data['close'] < midpoint)

            if volume_condition and candle_condition and low_open_condition and recover_condition:
                print(f"条件6触发：排除{code}，涨停日{day}")
                continue

        valid_days.append(day)
    return valid_days


def generate_signals(df, first_limit_day, stock_code, stock_name):
    # 首板收盘价，最重要的位置，表示主力的支撑度
    base_price = df.loc[first_limit_day, 'close']
    df['5ma'] = df['close'].rolling(5, min_periods=1).mean()
    post_limit_df = df[df.index >= first_limit_day].copy()

    # 条件1：涨停后每日收盘价高于首板收盘价（不包含首板日）
    always_above_base = True
    for date in post_limit_df.index:
        if date == first_limit_day:  # 跳过首板日（不需要比较）
            continue
        close_price = post_limit_df.loc[date, 'close']
        if close_price < base_price:
            always_above_base = False
            break

    # 判断每日是否触及五日线
    never_touched = True

    # 条件2：涨停日及之后每日未触及五日线
    for date in post_limit_df.index:
        if date == first_limit_day:  # 跳过首板日（不需要比较）
            continue
        row = post_limit_df.loc[date]
        # 判断是否触及：当日最低价 ≤ 五日线 ≤ 当日最高价
        if row['low'] <= row['5ma'] <= row['high']:
            never_touched = False
            break  # 一旦触及即退出循环

    return never_touched and always_above_base
    # return always_above_base


def get_target_stocks(isNeedLog=True):
    """获取目标股票列表，若当日数据已存在则直接读取"""
    # 检查当日数据是否已存在
    base_path = "output"
    file_path = os.path.join(base_path, "target_stocks_daily.csv")
    # 获取当前完整时间信息
    current_datetime = datetime.now()
    current_date_str = current_datetime.strftime('%Y-%m-%d')
    current_time = current_datetime.time()

    # 定义交易时段 9:31-15:00 交易时间内
    trading_start = time(9, 31)
    trading_end = time(15, 0)

    # 检查文件是否存在且包含当日数据
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        # 提取数据中的日期部分
        existing_dates = existing_df['日期'].apply(lambda x: x.split()[0])
        # 存在当日数据且处于交易时段
        if current_date_str in existing_dates.unique():
            # 判断当前是否在交易时段内
            if trading_start <= current_time <= trading_end:
                # 获取当日最新记录（按时间倒序）
                today_records = existing_df[existing_dates == current_date_str]
                today_latest = today_records.iloc[-1]

                stocks_str = today_latest['目标股票']
                target_stocks = stocks_str.split(',')
                print(f"交易时段直接读取当日数据：{len(target_stocks)}只股票")
                return target_stocks

    # ========== 当无当日数据时执行 ==========
    excluded_stocks = set()
    limit_up_stocks = []
    query_tool = getAllStockCsv.StockQuery()
    filtered_stocks = query_tool.get_all_filter_stocks()
    stock_list = filtered_stocks[['stock_code', 'stock_name']].values

    for idx, (code, name) in enumerate(stock_list, 1):
        df, _ = get_stock_data(code, isNeedLog)
        if df.empty:
            continue

        theme = query_tool.get_theme_by_code(code)
        # 买入距离涨停板3天内的票（越近胜率越高）
        first_limit_days = find_recent_first_limit_up(code, df, days=3)
        for day in first_limit_days:
            if generate_signals(df, day, code, name):
                limit_up_stocks.append((code, name, day.strftime("%Y-%m-%d"), theme))

    # 分组排序逻辑
    today = datetime.now().date()
    days_groups = {}
    print(f"\n总计发现 {len(limit_up_stocks)} 只符合要求的股票")

    for stock in limit_up_stocks:
        code, name, limit_date, theme = stock  # 拆包对象

        # 排除板块
        # if "光伏" in theme:  # 因为其他账户有大仓位光伏
        #     excluded_stocks.add(code)
        #     continue
        if "证券" in theme:  # 牛市旗手，跟不上，不参与
            excluded_stocks.add(code)
            continue
        # if "半导体" in theme:  # 因为其他账户有大仓位半导体，中芯和三安
        #     excluded_stocks.add(code)
        #     continue
        if "石油" in theme:  # 受海外消息影响过于严重，不参与
            excluded_stocks.add(code)
            continue
        if "外贸" in theme:  # 之前被恶心过
            excluded_stocks.add(code)
            continue

        # if "sh605378"==code:
        #     excluded_stocks.add(code)
        #     continue
        # if "sh600113"==code:
        #     excluded_stocks.add(code)
        #     continue
        # if "sh600113"==code:
        #     excluded_stocks.add(code)
        #     continue

        limit_day = datetime.strptime(limit_date, "%Y-%m-%d").date()
        delta_days = (today - limit_day).days
        if delta_days not in days_groups:
            days_groups[delta_days] = []
        days_groups[delta_days].append(stock)

    target_stocks = set()
    sorted_days = sorted(days_groups.items(), key=lambda x: x[0], reverse=False)

    for delta, stocks in sorted_days:
        for stock in stocks:
            code, name, date, theme = stock
            standard_code = getAllStockCsv.convert_to_standard_format(code)
            target_stocks.add(standard_code)
            print("  " + "   ".join(stock) + "  ")

    # 保存并返回新计算的数据
    save_target_stocks(target_stocks, excluded_stocks)
    return list(target_stocks)


def save_target_stocks(target_stocks, excluded_stocks, base_path="output"):
    """保存目标股票列表到CSV文件（股票代码按数字部分升序排序）"""
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, "target_stocks_daily.csv")
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M')

    # 格式化排除名单
    excluded_str = ",".join(sorted(
        excluded_stocks,
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )) if excluded_stocks else "无"

    sorted_stocks = sorted(
        target_stocks,
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    stocks_str = ",".join(sorted_stocks)

    today_df = pd.DataFrame({
        "日期": [current_date],
        "目标股票": [stocks_str],
        "排除股票": [excluded_str]
    })

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        if current_date in existing_df['日期'].values:
            existing_df.loc[existing_df['日期'] == current_date, '目标股票'] = stocks_str
            operation = "更新"
        else:
            existing_df = pd.concat([existing_df, today_df], ignore_index=True)
            operation = "添加"
        existing_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"已{operation}排序后的数据到: {file_path}")
    else:
        today_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"已创建新文件并保存排序后的数据到: {file_path}")

    return file_path


if __name__ == '__main__':
    # 获取目标股票列表
    target_stocks = get_target_stocks()

    # 打印结果
    print("\n目标股票列表:")
    for stock in target_stocks:
        print(stock)
    print(f"\n总数: {len(target_stocks)}只股票")