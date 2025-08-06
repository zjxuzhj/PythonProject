import os
import sys
from datetime import datetime, time

import pandas as pd

import first_limit_up_ma5_normal as normal
import getAllStockCsv
from first_limit_up_ma5_normal import StrategyConfig

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


def find_recent_first_limit_up(code, old_df):
    """识别最近days个交易日内存在的首板涨停日并排除连板"""
    config = StrategyConfig()
    limit_rate = config.MARKET_LIMIT_RATES[normal.get_market_type(code)]

    # 获取当前数据的最新日期
    end_date = old_df.index.max()
    if end_date is None or old_df.empty:
        return []

    # 计算涨停价
    old_df['prev_close'] = old_df['close'].shift(1)
    old_df['limit_price'] = (old_df['prev_close'] * (1 + limit_rate)).round(2)
    old_df['is_limit'] = old_df['close'] >= old_df['limit_price']

    # 筛选有效时间范围
    extended_days = 20
    start_date = (end_date - pd.offsets.BDay(extended_days)).strftime("%Y%m%d")
    date_mask = (old_df.index >= start_date) & (old_df.index <= end_date)
    df = old_df.loc[date_mask].copy()

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

        if normal.is_valid_first_limit_up_day(df, day, code, config, query_tool):
            valid_days.append(day)

    return valid_days


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
            print(f"---  进入回测模式，目标日期: {today.strftime('%Y-%m-%d')} ---")
    else:
        # --- 实时模式下的缓存逻辑 ---
        today = datetime.now().date()
        today_str = today.strftime('%Y-%m-%d')
        base_path, file_path = "output", os.path.join("output", "target_stocks_daily.csv")
        current_datetime, current_time = datetime.now(), datetime.now().time()

        if os.path.exists(file_path) and time(9, 31) <= current_time <= time(15, 0):
            existing_df = pd.read_csv(file_path)
            existing_dates = existing_df['日期'].apply(lambda x: x.split()[0])
            if today_str in existing_dates.values:
                latest_today_record = existing_df[existing_dates == today_str].iloc[-1]
                target_stocks = latest_today_record['目标股票'].split(',') if latest_today_record[
                                                                                  '目标股票'] != '无' else []
                fourth_day_stocks = latest_today_record['第四天股票'].split(',') if latest_today_record[
                                                                                        '第四天股票'] != '无' else []
                print(f"交易时段内，直接从缓存文件读取当日数据: {len(target_stocks)}只股票")
                return target_stocks, fourth_day_stocks
        # --- 缓存逻辑结束 ---

    if not is_backtest:
        print("--- 执行策略验证安全锁 ---")
        VALIDATION_DATE = "20250728"
        EXPECTED_STOCKS = {
            "000970.SZ", "002889.SZ", "600114.SH",
            "600410.SH", "603109.SH", "603630.SH"
        }

        # 以静默模式运行验证回测
        validation_stocks, _ = get_target_stocks(isNeedLog=False, target_date=VALIDATION_DATE)
        # 比较实际结果与预期结果
        if set(validation_stocks) != EXPECTED_STOCKS:
            print("\nX 策略验证失败！程序已终止。X")
            print("=" * 50)
            print(f"说明：当前策略在固定回测日 {VALIDATION_DATE} 的选股结果与预期不符。")
            print("这可能意味着您无意中修改了核心选股逻辑。为保证实盘安全，程序已停止运行。")
            print(f"预期结果 ({len(EXPECTED_STOCKS)}只): {sorted(list(EXPECTED_STOCKS))}")
            print(f"实际结果 ({len(validation_stocks)}只): {sorted(validation_stocks)}")
            print("=" * 50)
            sys.exit()  # 验证失败，终止程序
        else:
            print(f"策略验证通过！{VALIDATION_DATE} 的回测结果与预期一致。")
            print("--- ▶ 安全锁解除，开始执行今日任务 ---\n")

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

        first_limit_days = find_recent_first_limit_up(code, df)
        for day in first_limit_days:
            base_day_idx = df.index.get_loc(day)
            offset = len(df) - base_day_idx
            df['ma5'] = df['close'].rolling(5, min_periods=1).mean()
            if normal.is_valid_buy_opportunity(df, base_day_idx, offset, code, StrategyConfig()):
                theme = query_tool.get_theme_by_code(code)
                limit_up_stocks.append((code, name, day.strftime("%Y-%m-%d"), theme))

    days_groups = {}
    if isNeedLog: print(f"\n总计发现 {len(limit_up_stocks)} 只符合初步要求的股票")

    for code, name, limit_date_str, theme in limit_up_stocks:
        # 排除特定板块和股票
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
    # target_date = "20250731"
    # target_stocks, fourth_day_stocks = get_target_stocks(target_date=target_date)

    # 打印结果
    print("\n目标股票列表:")
    for stock in target_stocks:
        print(stock)
    print(f"\n总数: {len(target_stocks)}只股票")

    print("\n第四天目标股票列表:")
    for stock in fourth_day_stocks:
        print(stock)
    print(f"\n第四天总数: {len(fourth_day_stocks)}只股票")
