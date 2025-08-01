import os
import sys
from datetime import datetime, time

import pandas as pd

import getAllStockCsv

query_tool = getAllStockCsv.StockQuery()


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
    # recent_days_mask = (df.index > (end_date - pd.offsets.BDay(days)).strftime("%Y%m%d")) & (df.index <= end_date)
    # 只筛选第二天和第四天）：
    target_days = [end_date - pd.offsets.BDay(1), end_date - pd.offsets.BDay(3)]
    recent_days_mask = df.index.isin(target_days)

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

        # 条件-1 排除特定题材
        theme = query_tool.get_theme_by_code(code)
        name = query_tool.get_name_by_code(code)
        if "证券" in name or "金融" in name or "证券" in theme or "金融" in theme:  # 牛市旗手，跟不上，不参与
            continue
        if "石油" in name or "油气" in name or "石油" in theme:  # 受海外消息影响过于严重，不参与
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
                continue
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
        # day_idx = df.index.get_loc(day)
        # if day_idx >= 20:  # 确保20日历史数据
        #     # 计算前高（20日最高价）
        #     historical_high = df.iloc[day_idx - 20:day_idx]['high'].max()
        #     # 检查涨停前3日最高价是否触及前高的95%，获取涨停日前4个交易日（包括涨停日前3天、前2天、前1天，即索引位置day_idx-3到day_idx-1）
        #     recent_4day_high = df.iloc[day_idx - 4:day_idx]['high'].max()
        #     if historical_high * 0.95 <= recent_4day_high < historical_high:
        #         continue  # 触发排除条件

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
                body_ratio = (first_day_data['close'] - first_day_data['open']) / price_range
                candle_condition = (body_ratio > 0.5) and (first_day_data['close'] > first_day_data['open'])

            # 条件6-2：第二日低开且未收复前日实体中点
            # 计算实体中点：开盘价和收盘价的平均（因为是阳线）
            midpoint = (first_day_data['open'] + first_day_data['close']) / 2  # 前日阳线实体中点
            low_open_condition = (second_day_data['open'] < first_day_data['close'])  # 低开
            recover_condition = (second_day_data['close'] < midpoint)  # 盘中最高点未达中点

            if volume_condition and candle_condition and low_open_condition and recover_condition:
                print(f"条件6触发：排除{code}，涨停日{day}")
                continue

        # 条件7：排除市值大于250亿的股票
        market_value = query_tool.get_stock_market_value(code)
        if market_value > 250:
            continue

        # 条件8 - 排除10日内涨停次数过多的股票
        lookback_period_9 = 10
        day_idx = df.index.get_loc(day)
        if day_idx >= lookback_period_9:
            lookback_data_9 = df.iloc[day_idx - lookback_period_9: day_idx]
            limit_up_count = (lookback_data_9['close'] >= lookback_data_9['limit_price']).sum()
            if limit_up_count >= 4:
                continue

        # 条件10：排除10日内存在跌破一半的涨停
        lookback_period_10 = 4
        if day_idx >= lookback_period_10:
            lookback_data_10 = df.iloc[day_idx - lookback_period_10: day_idx]
            recent_limit_ups = lookback_data_10[lookback_data_10['close'] >= lookback_data_10['limit_price']]
            if not recent_limit_ups.empty:
                last_limit_up_day = recent_limit_ups.index[-1]
                # 获取最近涨停日的前一日收盘价
                prev_close_of_last_limit_up = df.loc[last_limit_up_day, 'prev_close']
                price_floor = prev_close_of_last_limit_up * 1.035
                intermediate_days_loc = slice(df.index.get_loc(last_limit_up_day) + 1, day_idx)
                intermediate_days = df.iloc[intermediate_days_loc]

                if not intermediate_days.empty:
                    min_low_in_between = intermediate_days['low'].min()
                    if min_low_in_between < price_floor:
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

    # return never_touched and always_above_base
    return always_above_base


def get_target_stocks(isNeedLog=True, target_date=None):
    """
    获取目标股票列表。
    - 如果 target_date 为 None (默认), 则为当日实时模式，会先进行策略验证，通过后才执行。
    - 如果提供了 target_date (e.g., "20250728"), 则为回测模式，不使用缓存或保存结果。
    """
    is_backtest = target_date is not None

    if is_backtest:
        today = datetime.strptime(target_date, '%Y%m%d').date()
        if isNeedLog:
            print(f"--- 🚀 进入回测模式，目标日期: {today.strftime('%Y-%m-%d')} ---")
    else:
        print("--- 🔍 执行策略验证安全锁 ---")
        VALIDATION_DATE = "20250728"
        EXPECTED_STOCKS = {
            "000970.SZ", "002889.SZ", "600114.SH",
            "600410.SH", "603109.SH", "603630.SH"
        }

        # 以静默模式运行验证回测
        validation_stocks, _ = get_target_stocks(isNeedLog=False, target_date=VALIDATION_DATE)
        # 比较实际结果与预期结果
        if set(validation_stocks) != EXPECTED_STOCKS:
            print("\n❌ 策略验证失败！程序已终止。❌")
            print("=" * 50)
            print(f"说明：当前策略在固定回测日 {VALIDATION_DATE} 的选股结果与预期不符。")
            print("这可能意味着您无意中修改了核心选股逻辑。为保证实盘安全，程序已停止运行。")
            print(f"预期结果 ({len(EXPECTED_STOCKS)}只): {sorted(list(EXPECTED_STOCKS))}")
            print(f"实际结果 ({len(validation_stocks)}只): {sorted(validation_stocks)}")
            print("=" * 50)
            sys.exit()  # 验证失败，终止程序
        else:
            print(f"✅ 策略验证通过！{VALIDATION_DATE} 的回测结果与预期一致。")
            print("--- ▶️ 安全锁解除，开始执行今日任务 ---\n")

        # --- 实时模式下的缓存逻辑 ---
        today = datetime.now().date()
        base_path, file_path = "output", os.path.join("output", "target_stocks_daily.csv")
        current_datetime, current_time = datetime.now(), datetime.now().time()

        if os.path.exists(file_path) and time(9, 31) <= current_time <= time(15, 0):
            existing_df = pd.read_csv(file_path)
            existing_dates = existing_df['日期'].apply(lambda x: x.split()[0])
            if today in existing_dates.values:
                latest_today_record = existing_df[existing_dates == today].iloc[-1]
                target_stocks = latest_today_record['目标股票'].split(',') if latest_today_record[
                                                                                  '目标股票'] != '无' else []
                fourth_day_stocks = latest_today_record['第四天股票'].split(',') if latest_today_record[
                                                                                        '第四天股票'] != '无' else []
                print(f"✅ 交易时段内，直接从缓存文件读取当日数据: {len(target_stocks)}只股票")
                return target_stocks, fourth_day_stocks
        # --- 缓存逻辑结束 ---

    # --- 数据处理核心逻辑 ---
    excluded_stocks, limit_up_stocks = set(), []
    stock_list = query_tool.get_all_filter_stocks()[['stock_code', 'stock_name']].values

    for code, name in stock_list:
        df, _ = get_stock_data(code, isNeedLog)
        if df.empty: continue

        if is_backtest:
            df = df[df.index < pd.Timestamp(today)]
            if df.empty: continue

        if pd.isna(df["close"].iloc[-1]):
            if isNeedLog:
                print(f"股票{code}最新收盘价为NaN（可能停牌或数据问题），跳过")
            continue
        # 排除当前股价>90的股票
        latest_close = df.iloc[-1]['close']
        if latest_close > 90:
            continue

        first_limit_days = find_recent_first_limit_up(code, df, days=4)
        for day in first_limit_days:
            if generate_signals(df, day, code, name):
                theme = query_tool.get_theme_by_code(code)
                limit_up_stocks.append((code, name, day.strftime("%Y-%m-%d"), theme))

    days_groups = {}
    if isNeedLog: print(f"\n🔍 总计发现 {len(limit_up_stocks)} 只符合初步要求的股票")

    for code, name, limit_date_str, theme in limit_up_stocks:
        # 排除特定板块和股票
        if any(exclude in theme for exclude in ["证券", "白酒", "石油", "外贸"]):
            excluded_stocks.add(getAllStockCsv.convert_to_standard_format(code))
            continue
        if code in ["sz002506", "sz002153"]:
            excluded_stocks.add(getAllStockCsv.convert_to_standard_format(code))
            continue

        limit_day = datetime.strptime(limit_date_str, "%Y-%m-%d").date()
        delta_days = (today - limit_day).days
        days_groups.setdefault(delta_days, []).append((code, name, limit_date_str, theme))

    target_stocks_set, fourth_day_stocks_set = set(), set()
    for delta, stocks in sorted(days_groups.items()):
        for stock_data in stocks:
            code = getAllStockCsv.convert_to_standard_format(stock_data[0])
            target_stocks_set.add(code)
            if isNeedLog: print("  " + "   ".join(stock_data) + "  ")

    # ===== 提取涨停后第四天的股票(delta_days=3) =====
    if 6 in days_groups:
        for stock_data in days_groups[6]:
            fourth_day_stocks_set.add(getAllStockCsv.convert_to_standard_format(stock_data[0]))

    target_stocks_list = sorted(list(target_stocks_set))
    fourth_day_stocks_list = sorted(list(fourth_day_stocks_set))

    if not is_backtest:
        save_target_stocks(target_stocks_list, excluded_stocks, fourth_day_stocks_list)

    return target_stocks_list, fourth_day_stocks_list


def save_target_stocks(target_stocks, excluded_stocks, fourth_day_stocks=None, base_path="output"):
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

    fourth_day_str = ",".join(sorted(
        fourth_day_stocks,
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )) if fourth_day_stocks else "无"

    today_df = pd.DataFrame({
        "日期": [current_date],
        "目标股票": [stocks_str],
        "第四天股票": [fourth_day_str],
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
    target_stocks, fourth_day_stocks = get_target_stocks()
    #
    # target_date = "20250728"
    # target_stocks,fourth_day_stocks = get_target_stocks(target_date=target_date)


    # 打印结果
    print("\n目标股票列表:")
    for stock in target_stocks:
        print(stock)
    print(f"\n总数: {len(target_stocks)}只股票")

    print("\n第四天目标股票列表:")
    for stock in fourth_day_stocks:
        print(stock)
    print(f"\n第四天总数: {len(fourth_day_stocks)}只股票")
