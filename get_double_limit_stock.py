import os
from datetime import datetime, timedelta

import pandas as pd

import getAllStockCsv


# import matplotlib.pyplot as plt

def interval_statistics(df, limit_rate):
    """统计间隔1-8天的涨停概率"""
    # 识别所有涨停日
    df['prev_close'] = df['close'].shift(1)
    df['limit_price'] = (df['prev_close'] * (1 + limit_rate)).round(2)
    limit_days = df[df['close'] >= df['limit_price']].index.tolist()

    # 初始化统计容器
    stats = {i: {"success": 0, "total": 0} for i in range(1, 9)}

    for i in range(len(limit_days) - 1):
        base_date = limit_days[i]
        next_limit = limit_days[i + 1]

        # 计算间隔天数（剔除非交易日）
        interval = (df.index.get_loc(next_limit) - df.index.get_loc(base_date) - 1)

        if 1 <= interval <= 8:
            stats[interval]["success"] += 1
            # 统计该间隔天数内其他未成功的情况
            for day in range(1, interval):
                stats[day]["total"] += 1  # 未成功计数

    # 计算概率
    result = []
    for day in stats:
        total = stats[day]["total"] + stats[day]["success"]
        success_rate = round(stats[day]["success"] / total * 100, 2) if total > 0 else 0
        result.append({
            "间隔天数": day,
            "成功次数": stats[day]["success"],
            "失败次数": total - stats[day]["success"],
            "成功比例": f"{success_rate}%",
            "失败比例": f"{100 - success_rate}%"
        })
    return pd.DataFrame(result)

def find_double_limit_up(symbol, df, isBackTest=False):
    """识别涨停双响炮形态"""
    # 获取市场类型及涨停规则
    market_type = "科创板" if symbol.startswith(("688", "689")) else "创业板" if symbol.startswith(
        ("300", "301")) else "主板"
    limit_rate = 0.20 if market_type in ["创业板", "科创板"] else 0.10

    # 识别所有涨停日
    df['prev_close'] = df['close'].shift(1)
    df['limit_price'] = (df['prev_close'] * (1 + limit_rate)).round(2)
    limit_days = df[df['close'] >= df['limit_price']].index.tolist()

    # 形态匹配
    valid_pairs = []
    for i in range(1, len(limit_days)):
        start_date = limit_days[i - 1]
        end_date = limit_days[i]
        # 计算间隔天数
        interval = df.index.get_loc(end_date) - df.index.get_loc(start_date) - 1

        # 新增：获取次板次日收盘价
        try:
            next_day = df.index[df.index.get_loc(end_date) + 1]  # 次板后一交易日
            next_close = df.loc[next_day, 'close']
            end_close = df.loc[end_date, 'close']
            next_pct = (next_close / end_close - 1) * 100  # 次日涨幅百分比
        except IndexError:
            next_pct = None  # 次板为最新交易日时无数据

        # 新增量比计算逻辑（参考网页4）
        start_volume = df.loc[start_date, 'volume']
        pre_volume = df.loc[df.index[df.index.get_loc(start_date) - 1], 'volume']
        volume_ratio = round(start_volume / pre_volume, 2)

        # 形态验证
        if 1 <= interval <= 10:
            # 获取中间K线数据
            mid_df = df[(df.index > start_date) & (df.index < end_date)]
            # 调整期最低价不低于首板最低价
            if mid_df['close'].min() >= df.loc[start_date]['high']:
            # if mid_df['close'].min() >= df.loc[start_date]['low']:
                valid_pairs.append((start_date, end_date, next_pct, volume_ratio))
    return valid_pairs


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


# 命令行显示效果
def get_double_limit_text(result_df, limit_up_stocks):
    if limit_up_stocks:
        # 新增首板放量统计（参考网页3）
        result_df['放量标记'] = result_df['量比'] >= 1  # 1.5倍为标准

        # 数据清洗（网页3方法）
        result_df['有效涨幅'] = pd.to_numeric(
            result_df['次日涨幅'].str.replace('%', ''),
            errors='coerce'
        )

        # 分组统计（网页7标准）
        interval_stats = result_df.groupby('间隔天数').agg(
            出现次数=('量比', 'count'),
            平均涨幅=('有效涨幅', lambda x: round(x.mean(), 2)),
            上涨概率=('有效涨幅', lambda x: round((x > 0).sum() / len(x) * 100, 1)),
            首板放量率=('放量标记', lambda x: f"{round(x.mean() * 100, 1)}%")
        ).reset_index()

        # 过滤有效数据（网页4要求）
        interval_stats = interval_stats[
            (interval_stats['间隔天数'].between(1, 10)) &
            (interval_stats['出现次数'] >= 3)  # 最小样本量要求
            ].sort_values('平均涨幅', ascending=False)

        # 生成详细分析报表
        print("\n\033[1m=== 不同间隔天数表现分析 ===\033[0m")
        print(interval_stats)

        # 最佳间隔判定
        best_interval = interval_stats.iloc[0]['间隔天数']
        print(f"\n\033[32m最优间隔天数：{best_interval}天（平均涨幅{interval_stats.iloc[0]['平均涨幅']}%）\033[0m")


# 用excel显示出现涨停双响炮得股票日期
def get_double_limit_excel(result_df):
    valid_pct = pd.to_numeric(result_df['次日涨幅'].str.replace('%', ''), errors='coerce')
    avg_pct = valid_pct.mean()

    # 计算平均间隔天数并添加统计行
    avg_days = result_df['间隔天数'].astype(float).mean()
    stats_row = pd.DataFrame([["-", "统计", "-", "-",
                               f"{avg_days:.1f}天", f"{avg_pct:.2f}%"]],
                             columns=result_df.columns)
    result_df = pd.concat([result_df, stats_row], ignore_index=True)

    # 保存到Excel（使用网页1、3、6的推荐方法）
    excel_name = f"double_limit_results_{datetime.now().strftime('%Y%m%d')}.xlsx"
    with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
        result_df.to_excel(writer, sheet_name='双响炮列表', index=False)

        # 获取工作表对象设置格式（网页3方法）
        workbook = writer.book
        worksheet = writer.sheets['双响炮列表']

        red_format = workbook.add_format({'font_color': '#FF0000'})
        green_format = workbook.add_format({'font_color': '#00B050'})

        # 设置涨幅列颜色（红跌绿涨）
        worksheet.conditional_format(
            f'F2:F{len(result_df) + 1}', {
                'type': 'cell',
                'criteria': '>=',
                'value': 0,
                'format': green_format
            })
        worksheet.conditional_format(
            f'F2:F{len(result_df) + 1}', {
                'type': 'cell',
                'criteria': '<',
                'value': 0,
                'format': red_format
            })

        # 设置标题格式
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        for col_num, value in enumerate(result_df.columns):
            worksheet.write(0, col_num, value, header_format)

        # 自动调整列宽（网页8方法）
        for idx, col in enumerate(result_df):
            series = result_df[col]
            max_len = max(series.astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(idx, idx, max_len)

    print(f"\033[32m数据已保存至 {excel_name}，平均间隔天数：{avg_days:.1f}天\033[0m")


def new_feature(symbol, df):
    """新增功能核心逻辑"""
    # 获取市场类型
    market_type = "科创板" if symbol.startswith(("688", "689")) else "创业板" if symbol.startswith(
        ("300", "301")) else "主板"
    limit_rate = 0.20 if market_type in ["创业板", "科创板"] else 0.10

    # 识别所有涨停日
    df['prev_close'] = df['close'].shift(1)
    df['limit_price'] = (df['prev_close'] * (1 + limit_rate)).round(2)
    limit_days = df[df['close'] >= df['limit_price']].index.tolist()

    results = []
    for i in range(len(limit_days)):
        base_date = limit_days[i]  # 首板日期
        base_close = df.loc[base_date, 'close']

        # 条件1：距离当前至少3个交易日[4](@ref)
        current_pos = df.index.get_loc(base_date)
        if (len(df) - current_pos) < 9:  # 至少需要3+6个交易日
            continue

        # 检查3日内未跌破首板收盘价
        check_period = df.iloc[current_pos + 1: current_pos + 4]  # 首板后3个交易日
        if (check_period['close'] < base_close).any():
            continue

        # 检查6日后的表现
        sixth_day = df.index[current_pos + 7]  # 首板日+7（含首板日）
        sixth_close = df.loc[sixth_day, 'close']
        is_break = 1 if sixth_close < base_close else 0

        results.append({
            '代码': symbol,
            '首板日': base_date.strftime('%Y-%m-%d'),
            '首板收盘价': base_close,
            '第六日收盘价': sixth_close,
            '是否跌破': is_break
        })
    return results


def calculate_probability(results):
    """计算跌破概率"""
    if not results:
        return 0.0

    total = len(results)
    break_count = sum(item['是否跌破'] for item in results)

    # 概率计算（保留两位小数）
    probability = round(break_count / total * 100, 2)

    # 生成统计报表（参考网页1/3的展示方式）
    stat_df = pd.DataFrame(results)
    print(f"\n\033[1m=== 跌破概率统计报表 ===\033[0m")
    print(f"总样本数：{total}")
    print(f"跌破次数：{break_count}")
    print(f"跌破概率：{probability}%")

    # 附加技术指标分析（参考网页2/6）
    if total > 0:
        print("\n\033[1m关联指标分析：\033[0m")
        print("• 可结合RSI指标判断超买状态[2](@ref)")
        print("• 建议观察MACD是否出现死叉[7](@ref)")

    return probability

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

    # 新增功能调用
    break_results = []

    for idx, (code, name) in enumerate(stock_list, 1):
        # try:
            df, _ = get_stock_data(code, start_date=start_date)
            if df.empty:
                continue

            break_results.extend(new_feature(code, df))
            double_limits = find_double_limit_up(code, df, isBackTest)
            for pair in double_limits:
                start, end, next_pct, volume_ratio  = pair
                # 获取交易日间隔（网页3标准）
                start_pos = df.index.get_loc(start)
                end_pos = df.index.get_loc(end)
                interval_days = end_pos - start_pos - 1  # 中间交易日数

                # 格式化涨幅显示
                pct_str = f"{next_pct:.2f}%" if next_pct is not None else "无数据"

                limit_up_stocks.append((
                    code, name,
                    f"首板:{start.strftime('%Y-%m-%d')}",
                    f"次板:{end.strftime('%Y-%m-%d')}",
                    interval_days,
                    pct_str,  # 新增涨幅列
                    volume_ratio  # 新增量比值
                ))

        # except Exception as e:
        #     print(f"处理异常：{name}({code}) - {str(e)}")
        #     continue

            # 创建带统计信息的DataFrame
    result_df = pd.DataFrame(limit_up_stocks, columns=[
        "代码", "名称", "首板日期", "次板日期", "间隔天数", "次日涨幅", "量比"
    ])

    get_double_limit_text(result_df, limit_up_stocks)

    # get_double_limit_excel(result_df)

    # calculate_probability(break_results)