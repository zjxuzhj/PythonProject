import os
from datetime import datetime, timedelta

import pandas as pd
import tabulate

import getAllStockCsv


def check_recent_limit_up(code, df, days=8):
    """检测最近days个交易日内是否有涨停且后续收盘价达标"""
    # 获取市场类型
    market = "科创板" if code.startswith(("688", "689")) else "创业板" if code.startswith(("300", "301")) else "主板"
    limit_rate = 0.20 if market in ["创业板", "科创板"] else 0.10
    # 新增动态时间范围计算（网页8）
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
    limit_days = recent_df[recent_df['close'] >= recent_df['limit_price']].index.tolist()

    for ld in limit_days:
        # 获取涨停日开盘价
        open_price = recent_df.loc[ld, 'open']
        # 检查后续所有收盘价是否达标
        subsequent_df = recent_df[recent_df.index > ld].head(8)  # 取后续最多8个交易日
        # 检测后续是否出现新涨停（网页2/5/7提到的RSI和资金面验证）
        subsequent_limit_days = subsequent_df[subsequent_df['close'] >= subsequent_df['limit_price']]
        if not subsequent_limit_days.empty:
            continue  # 排除有后续涨停的情况

        # 修改后（收盘价>涨停日最高价）
        # 新增涨停日最高价获取
        highest_price = recent_df.loc[ld, 'high']  # 获取涨停日最高价
        # 修改判断条件
        # if not subsequent_df.empty and (subsequent_df['close'] > highest_price).all():
        # if not subsequent_df.empty and (subsequent_df['close'] > open_price).all():
        # 确保比较运算生成完整的布尔序列
        if not subsequent_df.empty and (subsequent_df['close'] > open_price).all():
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

            # 调用新检测函数
            matched = check_recent_limit_up(code, df)
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
    sorted_days = sorted(days_groups.items(), key=lambda x: x[0])

    # 修改后的输出部分代码（替换原tabulate部分）
    headers = ["股票代码", "股票名称", "最近涨停日"]
    col_widths = [12, 16, 14]  # 列宽定义（单位：字符）

    for delta, stocks in sorted_days:
        # 打印分组标题
        print(f"\n\033[1;33m▲ {delta}天前首板股票 ▼\033[0m")  # 黄色加粗标题

        # 打印表头
        header_str = "".join([h.ljust(w) for h, w in zip(headers, col_widths)])
        print("\033[90m" + header_str + "\033[0m")  # 灰色表头

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
