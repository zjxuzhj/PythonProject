import os
from datetime import datetime, timedelta
import pandas as pd
import getAllStockCsv


def calculate_limit_count(df, days, market_type):
    """基于最后N条数据计算涨停次数（强制模式）"""
    if df.empty or days <= 0:
        return 0

    # 直接截取最后N条数据（网页1/2的核心方法）
    df_slice = df.tail(days).copy()

    # 计算涨停价（需处理首行prev_close为NaN的情况）
    df_slice['prev_close'] = df_slice['close'].shift(1)
    limit_rate = 0.20 if market_type in ["创业板", "科创板"] else 0.10
    df_slice['limit_price'] = (df_slice['prev_close'] * (1 + limit_rate)).round(2)

    # 筛选有效数据（排除首行因shift导致的NaN）
    valid_df = df_slice[df_slice['prev_close'].notna()]
    return (valid_df['close'] >= valid_df['limit_price']).sum()

def check_recent_limit_up(code, df, days=8, check_five_day_line=False):
    """检测最近days个交易日内是否有涨停且后续收盘价达标"""
    # 获取市场类型
    market = "科创板" if code.startswith(("688", "689")) else "创业板" if code.startswith(("300", "301")) else "主板"
    limit_rate = 0.20 if market in ["创业板", "科创板"] else 0.10
    if df.empty:
        return []

    # 获取当前数据的最新日期
    end_date = df.index.max()  # 数据的最新交易日
    # 计算days个交易日前的起始日期（网页1）
    start_date = (end_date - pd.offsets.BDay(days)).strftime("%Y%m%d")

    # 筛选有效时间范围（网页3）
    date_mask = (df.index >= start_date) & (df.index <= end_date)
    recent_df = df.loc[date_mask].copy()

    valid_stocks = []

    # 识别涨停日
    recent_df['prev_close'] = recent_df['close'].shift(1)
    recent_df['limit_price'] = (recent_df['prev_close'] * (1 + limit_rate)).round(2)
    recent_df['5ma'] = recent_df['close'].rolling(5, min_periods=5).mean()
    limit_days = recent_df[recent_df['close'] >= recent_df['limit_price']].index.tolist()

    for ld in limit_days:
        # 获取涨停日开盘价
        open_price = recent_df.loc[ld, 'open']
        # 检查后续所有收盘价是否达标
        subsequent_df = recent_df[recent_df.index > ld].head(8)  # 取后续最多8个交易日
        # 检测后续是否出现新涨停
        subsequent_limit_days = subsequent_df[subsequent_df['close'] >= subsequent_df['limit_price']]
        if not subsequent_limit_days.empty:
            continue  # 排除有后续涨停的情况

        # 新增空值检查
        if subsequent_df.empty:
            continue  # 跳过无后续数据的交易日
        subsequent_df['5ma'] = subsequent_df['close'].rolling(5, min_periods=1).mean()
        last_close = subsequent_df.iloc[-1]['close']
        last_date = subsequent_df.index[-1]  # 获取最后一个有效日期
        last_5ma = recent_df.loc[last_date, '5ma']  # 从完整序列中取对应日期的5日均线

        if check_five_day_line:  # True时保留五日线上方的股票
            if last_close <= last_5ma:
                continue
        else:  # False时排除五日线上方的股票
            if last_close > last_5ma:
                continue

        # 修改后（收盘价>涨停日最高价）
        # 新增涨停日最高价获取
        highest_price = recent_df.loc[ld, 'high']  # 获取涨停日最高价
        # 条件1：所有后续交易日收盘价 > 涨停日开盘价
        all_days_above_open = (subsequent_df['close'] > open_price).all()
        # 条件2：最后一天收盘价 < 涨停日最高价
        last_day_below_high = subsequent_df.iloc[-1]['close'] < highest_price
        subsequent_has_limit = (subsequent_df['close'] >= subsequent_df['limit_price']).any()

        # 判断条件
        if not subsequent_df.empty and (subsequent_df['close'] > highest_price).all():
        # if not subsequent_df.empty and (subsequent_df['close'] > open_price).all():
        # if all_days_above_open and last_day_below_high and not subsequent_has_limit:
            valid_stocks.append((code, name, ld.strftime("%Y-%m-%d")))

    return valid_stocks


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


isBackTest = False

if __name__ == '__main__':
    # 获取当日涨停数据（新增）
    today = datetime.now()
    today_str = today.strftime("%Y%m%d")
    yesterday = today - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y%m%d")

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
        try:
            df, _ = get_stock_data(code, start_date=start_date)
            if df.empty:
                continue

            # 判断市场类型（网页8阈值逻辑）
            market = "科创板" if code.startswith(("688", "689")) else "创业板" if code.startswith(
                ("300", "301")) else "主板"

            # 计算最近5天涨停次数（网页1核心逻辑）
            limit_count = calculate_limit_count(df, days=5, market_type=market)
            if limit_count >= 2:  # 网页5过滤条件
                print(f"跳过{name}({code}): 五天内涨停{limit_count}次")
                continue

            # 调用新检测函数
            matched = check_recent_limit_up(code, df, check_five_day_line=False)
            if matched:
                limit_up_stocks.extend(matched)


        except Exception as e:
            print(f"处理异常：{name}({code}) - {str(e)}")
            continue

    # 新增分组排序逻辑 ======================
    today = datetime.now().date()
    days_groups = {}

    print(f"\n总计发现 {len(limit_up_stocks)} 只符合要求的股票")

    for stock in limit_up_stocks:
        # 提取日期并转换为日期对象
        code, name, limit_date = stock
        limit_day = datetime.strptime(limit_date, "%Y-%m-%d").date()
        delta_days = (today - limit_day).days

        # 按天数分组
        if delta_days not in days_groups:
            days_groups[delta_days] = []
        days_groups[delta_days].append(stock)

    # 按天数排序（网页3排序方法）
    sorted_days = sorted(days_groups.items(), key=lambda x: x[0], reverse=False)

    # 修改后的输出部分代码（替换原tabulate部分）
    headers = ["股票代码", "股票名称", "最近涨停日"]
    col_widths = [12, 16, 14]  # 列宽定义（单位：字符）

    for delta, stocks in sorted_days:
        # 打印数据行
        for stock in stocks:
            code, name, date = stock
            # 动态调整名称显示长度（网页3/8方案）
            truncated_name = (name[:6] + "..") if len(name) > 8 else name.ljust(8)

            # 构建带对齐的输出行（网页5/7建议）
            line = f"{code.ljust(col_widths[0])}" \
                   f"{truncated_name.ljust(col_widths[1])}" \
                   f"{date.center(col_widths[2])}"

            print(line)

        # 打印分隔线（网页6增强可读性）
