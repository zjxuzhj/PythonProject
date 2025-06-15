import math
import os
from datetime import datetime
import time
import numpy as np
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
            # df['vol_ma5'] = df['volume'].rolling(5).mean()
            # df['volume_ratio'] = df['volume'] / df['vol_ma5'].replace(0, 1)  # 防除零错误
            print(f"从缓存加载数据：{symbol}")
            return df, True  # 返回缓存标记
        except Exception as e:
            print(f"缓存读取失败：{e}（建议删除损坏文件：{cache_path}）")

    print(f"数据获取失败：{symbol}")
    return pd.DataFrame()

def find_first_limit_up(symbol, df):
    """识别首板涨停日并排除连板"""
    market_type = "科创板" if symbol.startswith(("688", "689")) else "创业板" if symbol.startswith(
        ("300", "301")) else "主板"
    limit_rate = 0.20 if market_type in ["创业板", "科创板"] else 0.10
    # 计算涨停价
    df['prev_close'] = df['close'].shift(1)
    df['limit_price'] = (df['prev_close'] * (1 + limit_rate)).round(2)
    # 识别涨停日
    limit_days = df[df['close'] >= df['limit_price']].index.tolist()

    valid_days = []
    for day in limit_days:
        # 日期过滤条件（方便回测）
        if day < pd.Timestamp('2024-03-01'):
            continue

        # 条件1：排除连板（次日不涨停）
        next_day = df.index[df.index.get_loc(day) + 1] if (df.index.get_loc(day) + 1) < len(df) else None
        if next_day and df.loc[next_day, 'close'] >= df.loc[next_day, 'limit_price']:
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
            if next_day_change >=8:
                continue

        #  条件3：涨停后第一天量能过滤条件（放量存在出货可能）
        next_day_idx = df.index.get_loc(day) + 1
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
    """生成买卖信号"""
    signals = []
    first_limit_timestamp = pd.Timestamp(first_limit_day)
    market_type = "科创板" if stock_code.startswith(("688", "689")) else "创业板" if stock_code.startswith(
        ("300", "301")) else "主板"
    limit_rate = 0.20 if market_type in ["创业板", "科创板"] else 0.10

    base_price = df.loc[first_limit_day, 'close'] # 首板收盘价，最重要的位置，表示主力的支撑度
    df['down_limit_price'] = (df['prev_close'] * (1 - limit_rate)).round(2)  # 跌停价字段

    start_idx = df.index.get_loc(first_limit_day)
    next_next_day_change = None

    if start_idx + 2 < len(df):
        # 获取涨停日下一个交易日和下下个交易日的收盘价
        next_day = df.index[start_idx + 1]
        next_next_day = df.index[start_idx + 2]

        # 计算涨幅百分比（处理无效数据）
        base_price = df.loc[next_day, 'close']
        next_next_close = df.loc[next_next_day, 'close']

        # 确保收盘价有效（非零且非NaN）
        if not np.isnan(base_price) and not np.isnan(next_next_close) and base_price != 0:
            change_percent = (next_next_close - base_price) / base_price * 100

            # 安全处理可能的NaN
            if not np.isnan(change_percent):
                # 向上取整处理
                next_next_day_change = math.ceil(change_percent)  # 直接向上取整到整数
            else:
                print(f"{stock_code} 无效的涨幅计算: 基础价={base_price}, 下下日收盘价={next_next_close}")
        else:
            print(f"{stock_code} 无效的价格数据: 基础价={base_price}, 下下日收盘价={next_next_close}")

    if (start_idx + 1) >= len(df):  # 边界检查，跳过无数据的情况
        return signals
    # day1 = df.index[start_idx + 1]
    # if df.loc[day1, 'close'] < base_price: # 首板次日低于首板收盘价就跳过
    #     return signals

    # df['ma55'] = df['close'].rolling(60).mean()
    # df['ma30'] = df['close'].rolling(30).mean()
    df['ma5'] = df['close'].rolling(5).mean()

    first_touch_flag = False

    for offset in range(2, 10):  # 最多检查20个交易日
        if start_idx + offset >= len(df):
            break

        current_day = df.index[start_idx + offset]
        current_data = df.iloc[start_idx + offset]

        # 买入前收盘价不低于首板收盘价
        price_condition_met = True
        for check_offset in range(1, offset):
            check_day = df.index[start_idx + check_offset]
            if df.loc[check_day, 'close'] < base_price:
                price_condition_met = False
                break
        if not price_condition_met:
            continue

        # 获取最近5日MA5数据（防止空值）
        ma5_data = df['ma5'].iloc[start_idx:start_idx + offset + 1]
        if ma5_data.isnull().any():
            continue

        # 核心条件：当日最低价触碰五日均线
        touch_condition = (current_data['low'] <= current_data['ma5']) & \
                          (current_data['high'] >= current_data['ma5'])
        # tolerance = current_data['ma5'] * 0.002
        # touch_condition = (current_data['low'] <= current_data['ma5'] + tolerance) & \
        #                   (current_data['high'] >= current_data['ma5'])

        buy_day_timestamp = pd.Timestamp(current_day)
        days_after_limit = (buy_day_timestamp - first_limit_timestamp).days
        history_window = df.iloc[start_idx + 1: start_idx + offset]
        history_condition = (history_window['close'] > history_window['ma5']).all()

        if not first_touch_flag and touch_condition and history_condition:
            first_touch_flag = True  # 标记首次触碰
            buy_price = current_data['ma5']*1.02  # 以五日均线值为买入价
            hold_days = 0

            # 卖出逻辑
            for sell_offset in range(1, 20):  # 最多持有20日
                if start_idx + offset + sell_offset >= len(df):
                    break

                sell_day = df.index[start_idx + offset + sell_offset]
                sell_data = df.loc[sell_day]
                hold_days += 1

                # 1. 检查前一天是否跌停
                prev_day = df.index[df.index.get_loc(sell_day) - 1]
                prev_day_data = df.loc[prev_day]
                # 判断前一天是否跌停
                if prev_day_data['close'] <= prev_day_data['down_limit_price']:
                    # 第二天收盘价卖出
                    sell_price = sell_data['close']
                    profit_pct = (sell_price - buy_price) / buy_price * 100
                    signals.append({
                        '股票代码': stock_code,
                        '股票名称': stock_name,
                        '首板日': first_limit_day.strftime('%Y-%m-%d'),
                        '买入日': current_day.strftime('%Y-%m-%d'),
                        '卖出日': sell_day.strftime('%Y-%m-%d'),
                        '涨停后天数': days_after_limit,
                        '持有天数': hold_days,
                        '买入价': round(buy_price, 2),
                        '卖出价': round(sell_price, 2),
                        '触碰类型': 'MA5支撑反弹' if current_data['close'] > current_data['ma5'] else 'MA5破位回升',
                        '收益率(%)': round(profit_pct, 2),
                        '首板下下日涨幅(%)': next_next_day_change,
                        '卖出原因': '跌停止损'
                    })
                    break

                if sell_data['close'] >= sell_data['limit_price']:
                    continue  # 涨停日继续持有

                # 2. 断板日卖出（止盈条件）
                # 前一日涨停但当日未涨停（断板）
                prev_day = df.index[df.index.get_loc(sell_day) - 1]
                if df.loc[prev_day, 'close'] >= df.loc[prev_day, 'limit_price']:
                    profit_pct = (sell_data['close'] - buy_price) / buy_price * 100
                    signals.append({
                        '股票代码': stock_code,
                        '股票名称': stock_name,
                        '首板日': first_limit_day.strftime('%Y-%m-%d'),
                        '买入日': current_day.strftime('%Y-%m-%d'),
                        '卖出日': sell_day.strftime('%Y-%m-%d'),
                        '涨停后天数': days_after_limit,
                        '持有天数': hold_days,
                        '买入价': round(buy_price, 2),
                        '卖出价': round(sell_data['close'], 2),
                        '触碰类型': 'MA5支撑反弹' if current_data['close'] > current_data['ma5'] else 'MA5破位回升',
                        '收益率(%)': round(profit_pct, 2),
                        '首板下下日涨幅(%)': next_next_day_change,
                        '卖出原因': '断板止盈'  # 新增字段
                    })
                    break

                # 3. 跌破五日线卖出条件
                if sell_data['close'] < sell_data['ma5']:
                    profit_pct = (sell_data['close'] - buy_price) / buy_price * 100
                    signals.append({
                        '股票代码': stock_code,
                        '股票名称': stock_name,
                        '首板日': first_limit_day.strftime('%Y-%m-%d'),
                        '买入日': current_day.strftime('%Y-%m-%d'),
                        '卖出日': sell_day.strftime('%Y-%m-%d'),
                        '涨停后天数': days_after_limit,
                        '持有天数': hold_days,
                        '买入价': round(buy_price, 2),
                        '卖出价': round(sell_data['close'], 2),
                        '触碰类型': 'MA5支撑反弹' if current_data['close'] > current_data['ma5'] else 'MA5破位回升',
                        '收益率(%)': round(profit_pct, 2),
                        '首板下下日涨幅(%)': next_next_day_change,
                        '卖出原因': '跌破五日线'
                    })
                    break

                # 4. 最大持有天数限制（15天）
                if hold_days >= 15:
                    profit_pct = (sell_data['close'] - buy_price) / buy_price * 100
                    signals.append({
                        '股票代码': stock_code,
                        '股票名称': stock_name,
                        '首板日': first_limit_day.strftime('%Y-%m-%d'),
                        '买入日': current_day.strftime('%Y-%m-%d'),
                        '卖出日': sell_day.strftime('%Y-%m-%d'),
                        '涨停后天数': days_after_limit,
                        '持有天数': hold_days,
                        '买入价': round(buy_price, 2),
                        '卖出价': round(sell_data['close'], 2),
                        '触碰类型': 'MA5支撑反弹' if current_data['close'] > current_data['ma5'] else 'MA5破位回升',
                        '收益率(%)': round(profit_pct, 2),
                        '首板下下日涨幅(%)': next_next_day_change,
                        '卖出原因': '持有超限'
                    })
                    break
            break  # 只取第一次触碰
    return signals


def save_trades_excel(result_df):
    column_order = ['股票代码', '股票名称', '首板日', '买入日', '卖出日','涨停后天数',
                    '持有天数', '买入价', '卖出价', '收益率(%)','首板下下日涨幅(%)','卖出原因']
    # 按买入日降序排序
    result_df = result_df.sort_values(by='买入日', ascending=False)
    result_df = result_df[column_order]
    """专业级Excel导出函数"""
    # 生成带时间戳的文件名
    excel_name = f"首板交易记录_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

    # 创建带格式的Excel写入器
    with pd.ExcelWriter(excel_name, engine='xlsxwriter',engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
        # 写入原始数据
        result_df.to_excel(writer, sheet_name='交易明细', index=False)

        # 获取工作表对象
        workbook = writer.book
        worksheet = writer.sheets['交易明细']

        # 1. 列宽自适应
        for idx, col in enumerate(result_df.columns):
            series = result_df[col]
            max_len = max((
                series.astype(str).map(len).max(),  # 数据最大长度
                len(col)  # 列名长度
            )) + 2
            worksheet.set_column(idx, idx, max_len)

        # 2. 条件格式(收益率红涨绿跌)
        format_green = workbook.add_format({'font_color': '#00B050', 'num_format': '0.00%'})
        format_red = workbook.add_format({'font_color': '#FF0000', 'num_format': '0.00%'})

        # 将收益率转换为小数格式
        worksheet.write(0, result_df.columns.get_loc('收益率(%)'), '收益率',
                        workbook.add_format({'bold': True}))
        for row in range(1, len(result_df) + 1):
            cell_value = result_df.iloc[row - 1]['收益率(%)'] / 100
            if cell_value >= 0:
                worksheet.write(row, result_df.columns.get_loc('收益率(%)'), cell_value, format_green)
            else:
                worksheet.write(row, result_df.columns.get_loc('收益率(%)'), cell_value, format_red)

        # 3. 冻结首行
        worksheet.freeze_panes(1, 0)
        # 4. 自动筛选
        worksheet.autofilter(0, 0, len(result_df), len(result_df.columns) - 1)
        # 5. 添加统计页
        stats_df = pd.DataFrame({
            '统计指标': ['总交易次数', '胜率', '平均盈利', '平均亏损', '盈亏比'],
            '数值': [
                len(result_df),
                f"{result_df['收益率(%)'].gt(0).mean() * 100:.1f}%",
                f"{result_df[result_df['收益率(%)'] > 0]['收益率(%)'].mean():.1f}%",
                f"{result_df[result_df['收益率(%)'] <= 0]['收益率(%)'].mean():.1f}%",
                f"{result_df['收益率(%)'].gt(0).sum() / result_df['收益率(%)'].le(0).sum():.1f}:1"
            ]
        })
        stats_df.to_excel(writer, sheet_name='策略统计', index=False)
    print(f"\033[32m交易记录已保存至 {excel_name}\033[0m")


if __name__ == '__main__':
    total_start = time.perf_counter()  # 记录程序开始时间

    all_signals = []
    query_tool = getAllStockCsv.StockQuery()
    # 加载股票列表并过滤
    filtered_stocks = query_tool.get_all_filter_stocks()
    stock_list = filtered_stocks[['stock_code', 'stock_name']].values
    total = len(stock_list)
    stock_process_start = time.perf_counter()

    for idx, (code, name) in enumerate(stock_list, 1):
        df, _ = get_stock_data(code)
        if df.empty:
            continue

        first_limit_days = find_first_limit_up(code, df)
        for day in first_limit_days:
            signals = generate_signals(df, day, code, name)
            all_signals.extend(signals)

    stock_process_duration = time.perf_counter() - stock_process_start

    # 生成统计报表
    result_df = pd.DataFrame(all_signals)
    if not result_df.empty:
        win_rate = len(result_df[result_df['收益率(%)'] > 0]) / len(result_df) * 100
        avg_win = result_df[result_df['收益率(%)'] > 0]['收益率(%)'].mean()
        avg_loss = abs(result_df[result_df['收益率(%)'] <= 0]['收益率(%)'].mean())
        profit_ratio = avg_win / avg_loss if avg_loss != 0 else np.inf
        get_money=len(result_df[result_df['收益率(%)'] > 0]) / len(result_df)*avg_win-(1-len(result_df[result_df['收益率(%)'] > 0]) / len(result_df))*avg_loss

        print(f"\n\033[1m=== 策略表现汇总 ===\033[0m")
        print(f"总交易次数: {len(result_df)}")
        print(f"胜率: {win_rate:.2f}%")
        print(f"平均盈利: {avg_win:.2f}% | 平均亏损: {avg_loss:.2f}%")
        print(f"盈亏比: {profit_ratio:.2f}:1")
        print(f"期望收益: {get_money:.3f}")

        # 示例输出
        print("\n\033[1m最近5笔交易记录:\033[0m")
        print(result_df.tail(5).to_string(index=False))
    else:
        print("未产生有效交易信号")

    if not result_df.empty:
        save_start = time.perf_counter()  # 记录Excel保存开始时间
        save_trades_excel(result_df)
        save_duration = time.perf_counter() - save_start
        print(f"Excel保存耗时: {save_duration:.4f}秒")

    # 程序总耗时统计
    total_duration = time.perf_counter() - total_start
    print(f"\n\033[1m=== 性能统计 ===\033[0m")
    print(f"总运行时间: {total_duration:.2f}秒")
    print(f"股票数据处理时间: {stock_process_duration:.2f}秒")
    print(f"Excel保存时间: {save_duration:.4f}秒")
    print(f"平均每支股票处理时间: {stock_process_duration/len(stock_list)*1000:.2f}毫秒")