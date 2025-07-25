import os
from datetime import datetime, timedelta

import pandas as pd

import getAllStockCsv


def find_first_touch_after_limit(limit_up_stocks):
    """筛选最近涨停后首次触碰五日线的股票"""
    today = datetime.now().date()
    results = []

    # 按股票代码分组，获取最近一次涨停日
    stock_last_limit = {}
    for code, name, limit_date, industry in limit_up_stocks:
        limit_day = datetime.strptime(limit_date, "%Y-%m-%d").date()
        # 保留最近一次涨停日
        if code not in stock_last_limit or limit_day > stock_last_limit[code]['limit_date']:
            stock_last_limit[code] = {
                'name': name,
                'limit_date': limit_day,
                'industry': industry
            }

    for code, stock_info in stock_last_limit.items():
        name = stock_info['name']
        limit_date = stock_info['limit_date']

        df, _ = get_stock_data(code, start_date="20240201")

        if df.empty :
            continue

        # 定位涨停日在数据中的位置
        limit_idx = df.index.get_loc(pd.Timestamp(limit_date))
        if limit_idx < 0:
            continue

        # 计算五日线（从涨停日开始计算）
        df['ma5'] = df['close'].rolling(5).mean()

        # 检查涨停日后是否首次触碰五日线
        first_touch = True
        for i in range(limit_idx + 1, len(df) - 1):  # 排除今日
            day_data = df.iloc[i]
            if (day_data['low'] <= day_data['ma5']) and (day_data['high'] >= day_data['ma5']):
                first_touch = False
                break

        if not first_touch:
            continue

        # 检查今日是否触碰五日线
        today_data = df.iloc[-1]
        touch_condition = (today_data['low'] <= today_data['ma5']) & (today_data['high'] >= today_data['ma5'])

        if touch_condition:
            pct_diff = ((today_data['close'] - today_data['ma5']) / today_data['ma5']) * 100
            results.append({
                '股票代码': code,
                '股票名称': name,
                '涨停日期': limit_date.strftime("%Y-%m-%d"),
                '收盘价': round(today_data['close'], 2),
                '五日线': round(today_data['ma5'], 2),
                '百分比差(%)': round(pct_diff, 2),
                '行业': stock_info['industry']
            })

    # 按百分比差降序排序
    return sorted(results, key=lambda x: x['百分比差(%)'], reverse=True)

def calculate_down_limit_count(df, days, market_type):
    """计算最近days个交易日内跌停次数"""
    if df.empty or days <= 0:
        return 0

    # 截取最后N条数据（同网页1/2逻辑）
    df_slice = df.tail(days).copy()

    # 计算跌停价（注意创业板/科创板规则）
    down_limit_rate = 0.20 if market_type in ["创业板", "科创板"] else 0.10
    df_slice['prev_close'] = df_slice['close'].shift(1)
    df_slice['down_limit_price'] = (df_slice['prev_close'] * (1 - down_limit_rate)).round(2)

    # 筛选有效数据
    valid_df = df_slice[df_slice['prev_close'].notna()]
    return (valid_df['close'] <= valid_df['down_limit_price']).sum()


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
        # if not subsequent_df.empty and (((subsequent_df['close'] > open_price) &(subsequent_df['close'] < highest_price)).all()):
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

            # 检查最近6日跌停情况（网页4/5的跌停判断逻辑）
            down_limit_count = calculate_down_limit_count(df, days=5, market_type=market)
            if down_limit_count > 0:
                print(f"排除{name}({code}): 六天内出现{down_limit_count}次跌停")
                continue

            # 计算最近5天涨停次数（网页1核心逻辑）
            limit_count = calculate_limit_count(df, days=5, market_type=market)
            if limit_count >= 2:  # 网页5过滤条件
                print(f"跳过{name}({code}): 五天内涨停{limit_count}次")
                continue

            # 调用新检测函数
            matched = check_recent_limit_up(code, df, check_five_day_line=True)
            if matched:
                symbol = ('sh' + code if code.startswith(('6', '9', '688', '689'))
                          else 'sz' + code if code.startswith(('0', '3', '300', '301'))
                else code)
                # 获取行业信息（含异常处理）
                try:
                    industry = query_tool.get_stock_industry(symbol) or "未知行业"
                except:
                    industry = "行业获取失败"
                for item in matched:
                    code, name, date = item
                    limit_up_stocks.append((code, name, date, industry))

                # limit_up_stocks.extend(matched)


        except Exception as e:
            print(f"处理异常：{name}({code}) - {str(e)}")
            continue

    # 新增分组排序逻辑 ======================
    today = datetime.now().date()
    days_groups = {}

    print(f"\n总计发现 {len(limit_up_stocks)} 只符合要求的股票")

    touch_stocks = find_first_touch_after_limit(limit_up_stocks)

    if touch_stocks:
        print("\n=== 涨停后首次触碰五日线股票（按强度降序）===")
        headers = ["股票代码", "股票名称", "涨停日期", "收盘价", "五日线", "百分比差(%)", "行业"]
        col_widths = [10, 12, 12, 10, 10, 12, 20]

        # 打印表头
        header_line = "".join(header.ljust(width) for header, width in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))

        # 打印数据行
        for stock in touch_stocks:
            pct_str = f"{stock['百分比差(%)']:.2f}%"
            line = (
                f"{stock['股票代码'].ljust(col_widths[0])}"
                f"{stock['股票名称'].ljust(col_widths[1])}"
                f"{stock['涨停日期'].ljust(col_widths[2])}"
                f"{str(stock['收盘价']).rjust(col_widths[3])}"
                f"{str(stock['五日线']).rjust(col_widths[4])}"
                f"{pct_str.rjust(col_widths[5])}"
                f"{stock['行业'].ljust(col_widths[6])}"
            )
            print(line)
    else:
        print("\n今日无涨停后首次触碰五日线的股票")