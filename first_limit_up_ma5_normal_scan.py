import os
from datetime import datetime, timedelta

import pandas as pd

import getAllStockCsv


def check_recent_limit_up(code, df, days=8):
    """检测最近days个交易日内是否有涨停且后续收盘价达标"""
    # 获取市场类型
    market = "科创板" if code.startswith(("688", "689")) else "创业板" if code.startswith(("300", "301")) else "主板"
    limit_rate = 0.20 if market in ["创业板", "科创板"] else 0.10
    if df.empty:
        return []

    # 获取当前数据的最新日期
    end_date = df.index.max()  # 数据的最新交易日
    # 计算days个交易日前的起始日期
    start_date = (end_date - pd.offsets.BDay(days)).strftime("%Y%m%d")

    # 筛选有效时间范围
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
        # 检查后续所有收盘价是否达标
        subsequent_df = recent_df[recent_df.index > ld].head(8)  # 取后续最多8个交易日

        # 筛选条件：排除涨停后第一天涨幅≥8%
        if not subsequent_df.empty:
            first_day_after_limit = subsequent_df.iloc[0]
            prev_day_close = recent_df.loc[ld, 'close']
            day1_pct = (first_day_after_limit['close'] - prev_day_close) / prev_day_close * 100
            if day1_pct >= 8:
                continue

        # 条件一：排除连板
        subsequent_limit_days = subsequent_df[subsequent_df['close'] >= subsequent_df['limit_price']]
        if not subsequent_limit_days.empty:
            continue

        if subsequent_df.empty:
            continue  # 跳过无后续数据的交易日
        subsequent_df['5ma'] = subsequent_df['close'].rolling(5, min_periods=1).mean()
        last_close = subsequent_df.iloc[-1]['close']
        last_date = subsequent_df.index[-1]  # 获取最后一个有效日期
        last_5ma = recent_df.loc[last_date, '5ma']  # 从完整序列中取对应日期的5日均线

        if last_close <= last_5ma:
            continue

        base_price = recent_df.loc[ld, 'high']

        if not subsequent_df.empty and (subsequent_df['close'] > base_price).all():
            valid_stocks.append((code, name, ld.strftime("%Y-%m-%d")))

    return valid_stocks


def get_stock_data(symbol):
    """带本地缓存的数据获取"""
    # 生成唯一文件名（网页1）
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

    # 初始化涨停股容器
    limit_up_stocks = []

    query_tool = getAllStockCsv.StockQuery()
    filtered_stocks = query_tool.get_all_filter_stocks()
    stock_list = filtered_stocks[['stock_code', 'stock_name']].values
    total = len(stock_list)

    for idx, (code, name) in enumerate(stock_list, 1):
        try:
            df, _ = get_stock_data(code)
            if df.empty:
                continue

            matched = check_recent_limit_up(code, df)
            if matched:
                symbol = ('sh' + code if code.startswith(('6', '9', '688', '689'))
                          else 'sz' + code if code.startswith(('0', '3', '300', '301'))
                else code)
                for item in matched:
                    code, name, date_str = item
                    date = pd.Timestamp(date_str)

                    position = df.index.get_loc(date)
                    pre5_days_pct = "N/A"  # 默认值

                    if position >= 5:
                        pre5_day = df.index[position - 5]
                        pre5_close = df.loc[pre5_day, 'close']
                        limit_close = df.loc[date, 'close']
                        pre5_days_pct = (limit_close - pre5_close) / pre5_close * 100
                        if pre5_days_pct > 15:
                            continue

                    limit_up_stocks.append((code, name, date_str))

        except Exception as e:
            print(f"处理异常：{name}({code}) - {str(e)}")
            continue

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
