import os
import sys
from datetime import datetime, time

import pandas as pd
import first_limit_up_ma5_normal as normal
import getAllStockCsv
from first_limit_up_ma5_normal import StrategyConfig
from config.stock_exclusion_manager import StockExclusionManager
from stock_info import StockInfo

query_tool = getAllStockCsv.StockQuery()


def get_stock_data(symbol, isNeedLog):
    file_name = f"stock_{symbol}_20240201.parquet"
    root_dir = os.path.abspath(os.path.dirname(__file__))
    cache_path = os.path.join(root_dir, "data_cache", file_name)
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path, engine='fastparquet')
            if isNeedLog:
                print(f"从缓存加载数据：{symbol}")
            return df, True
        except Exception as e:
            print(f"缓存读取失败：{e}（建议删除损坏文件：{cache_path}）")
    print(f"数据获取失败：{symbol}")
    return pd.DataFrame(), False


def get_previous_trading_day(df: pd.DataFrame, end_date, n: int = 1):
    """基于本地数据的交易日索引，返回 end_date 之前的第 n 个交易日。

    - 先使用策略文件中的本地交易日过滤（移除周末/复制K线），统一交易日口径
    - 若 end_date 非交易日，则从最近一个不晚于 end_date 的交易日开始回溯
    - 返回 pd.Timestamp；若不足 n 天或数据异常，返回 None
    """
    try:
        if df is None or df.empty or n < 0:
            return None

        # 使用统一的本地交易日过滤，避免节假日填充导致的误判
        try:
            df = normal.filter_df_to_trading_days_local(df)
        except Exception:
            pass

        # 统一索引为时间类型并升序
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)

        end_ts = pd.Timestamp(end_date)
        unique_days = df.index.unique().sort_values()

        # 找到 <= end_date 的最近一个交易日位置
        pos = unique_days.searchsorted(end_ts, side='right') - 1
        if pos < 0:
            # end_date 早于最早数据，无法回溯
            return None

        target_pos = pos - n
        if target_pos < 0:
            return None

        return unique_days[target_pos]
    except Exception:
        return None


def find_recent_first_limit_up(stock_info: StockInfo, df):
    """识别最近days个交易日内存在的首板涨停日并排除连板"""
    config = StrategyConfig()

    # 获取当前数据的最新日期
    end_date = df.index.max()
    if end_date is None or df.empty:
        return []

    # 筛选有效时间范围
    # extended_days = 130
    # start_date = (end_date - pd.offsets.BDay(extended_days)).strftime("%Y%m%d")
    # date_mask = (old_df.index >= start_date) & (old_df.index <= end_date)
    # df = old_df.loc[date_mask].copy()
    # 基于交易日索引动态定位：涨停后第2天/第4天对应的涨停日应为 end_date 的前1/3个交易日
    d1 = get_previous_trading_day(df, end_date, 1)
    d3 = get_previous_trading_day(df, end_date, 3)
    target_days = [d for d in [d1, d3] if d is not None]
    if not target_days:
        return []
    recent_days_mask = df.index.isin(target_days)

    limit_days = df[df['is_limit'] & recent_days_mask].index.tolist()

    # 排除涨停日是最后一天的情况（若存在），但不因 last_day 涨停而整体退出
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

        rejection_rule = normal.is_valid_first_limit_up_day(stock_info, df, day, config)
        if not rejection_rule:
            valid_days.append(day)

    return valid_days


def get_target_stocks(isNeedLog=True, target_date=None):
    """
    获取目标股票列表。
    - 如果 target_date 为 None (默认), 则为当日实时模式，会先进行策略验证，通过后才执行。
    - 如果提供了 target_date (e.g., "20250728"), 则为回测模式，不使用缓存或保存结果。
    """
    is_backtest = target_date is not None
    config = StrategyConfig()  # 获取策略配置

    if is_backtest:
        today = datetime.strptime(target_date, '%Y%m%d').date()
        if isNeedLog:
            print(f"---  进入回测模式，目标日期: {today.strftime('%Y-%m-%d')} ---")
    else:
        # --- 实时模式下的缓存逻辑 ---
        today = datetime.now().date()
        today_str = today.strftime('%Y-%m-%d')
        base_path = os.path.join(os.path.dirname(__file__), "miniqmt_etf", "output")
        file_path = os.path.join(base_path, "target_stocks_daily.csv")
        current_datetime, current_time = datetime.now(), datetime.now().time()

        if os.path.exists(file_path) and time(9, 20) <= current_time <= time(15, 0):
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
            "002889.SZ", "603109.SH", "600410.SH",'600515.SH'
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

        # 统一使用本地交易日过滤，避免节假日填充/周末导致的偏移错误
        before_len = len(df)
        df = normal.filter_df_to_trading_days_local(df)
        after_len = len(df)
        # if isNeedLog and before_len != after_len:
        #     print(f"[{code}] 交易日过滤: 原始 {before_len} 天 -> 保留 {after_len} 天 -> 移除 {before_len - after_len} 天")

        if pd.isna(df["close"].iloc[-1]):
            if isNeedLog:
                print(f"股票{code}最新收盘价为NaN（可能停牌或数据问题），跳过")
            continue

        # --- 核心修改：采用优化后的处理流程 ---
        # 1. 一次性准备好所有数据（计算指标）
        df = normal.prepare_data(df, code, config)

        # 2. 一次性查询静态数据
        stock_info = StockInfo(
            code=code,
            name=name,
            market_value=query_tool.get_stock_market_value(code),
            theme=query_tool.get_theme_by_code(code),
        )
        first_limit_days = find_recent_first_limit_up(stock_info, df)

        for day in first_limit_days:
            base_day_idx = df.index.get_loc(day)
            # 使用从涨停日到最近一日(含)的交易日计数，直接映射为 T+N
            offset = len(df) - base_day_idx  # T+N 中的 N = offset

            rejection_rule = normal.is_valid_buy_opportunity(stock_info, df, base_day_idx, offset, StrategyConfig())
            if not rejection_rule:
                score, reasons = normal.calculate_quality_score(stock_info, df, base_day_idx, offset, config)
                theme = query_tool.get_theme_by_code(code)
                # 直接使用 offset 映射为 T+N，避免 off-by-one
                delta_days_actual = offset
                limit_up_stocks.append((code, name, day.strftime("%Y-%m-%d"), theme, score, delta_days_actual))

    days_groups = {}
    if isNeedLog: print(f"\n总计发现 {len(limit_up_stocks)} 只符合初步要求的股票")

    # 统一排除管理（单例，配置化）
    exclusion_manager = StockExclusionManager.get_instance()

    for code, name, limit_date_str, theme, score, delta_days in limit_up_stocks:
        # 统一的排除判断（名称/题材模式匹配 + 代码黑名单）
        try:
            if exclusion_manager.should_exclude(code, name, theme):
                excluded_stocks.add(getAllStockCsv.convert_to_standard_format(code))
                continue
        except Exception:
            # 排除判断异常不影响后续逻辑
            pass
        # 排除分数小于或等于10的股票
        # if score < 10:
        if score < 10:
            excluded_stocks.add(getAllStockCsv.convert_to_standard_format(code))
            continue
        days_groups.setdefault(delta_days, []).append((code, name, limit_date_str, theme, score))

    target_stocks_list = []
    fourth_day_stocks_list = []
    target_stocks_set, fourth_day_stocks_set = set(), set()
    for delta, stocks in sorted(days_groups.items()):
        # 对当前日期的股票列表按分数（第5个元素，索引4）降序排序
        stocks.sort(key=lambda x: x[4], reverse=True)
        for stock_data in stocks:
            code = getAllStockCsv.convert_to_standard_format(stock_data[0])
            target_stocks_set.add(code)
            target_stocks_list.append(code)
            if isNeedLog: print("  " + "   ".join(map(str, stock_data)) + "  ")

    # ===== 提取涨停后第四天的股票 (delta_days = 4) =====
    if 4 in days_groups:
        for stock_data in days_groups[4]:
            fourth_day_stocks_set.add(getAllStockCsv.convert_to_standard_format(stock_data[0]))

    target_stocks_list = sorted(list(target_stocks_set))
    fourth_day_stocks_list = sorted(list(fourth_day_stocks_set))

    if not is_backtest:
        save_target_stocks(target_stocks_list, excluded_stocks, fourth_day_stocks_list)

    excluded_str = ",".join(sorted(
        excluded_stocks,
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )) if excluded_stocks else "无"
    if isNeedLog: print(f"\n排除股票: {excluded_str}")

    return target_stocks_list, fourth_day_stocks_list


def save_target_stocks(target_stocks, excluded_stocks, fourth_day_stocks=None, base_path=None):
    """保存目标股票列表到CSV文件（股票代码按数字部分升序排序）"""
    if base_path is None:
        base_path = os.path.join(os.path.dirname(__file__), "miniqmt_etf", "output")
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
    # target_stocks, fourth_day_stocks = get_target_stocks()

    # 填入的指定日期是当天收盘的日期，如果需要明天的买入列表就填前一日的日期
    target_date = "20251124"
    target_stocks, fourth_day_stocks = get_target_stocks(target_date=target_date)

    # 打印结果    print("\n目标股票列表:")
    for stock in target_stocks:
        print(stock)
    print(f"\n总数: {len(target_stocks)}只股票")

    print("\n第四天目标股票列表:")
    for stock in fourth_day_stocks:
        print(stock)
    print(f"\n第四天总数: {len(fourth_day_stocks)}只股票")
