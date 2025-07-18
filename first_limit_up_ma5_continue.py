import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import getAllStockCsv

# 连板票，五日买入
# 总交易次数: 1533，胜率: 41.9%，平均盈利: 9.9% | 平均亏损: 5.5%，盈亏比: 1.79:1，期望收益: 0.94（连板股，断板止盈五日线买入逻辑）
def get_stock_data(symbol, start_date, force_update=False):
    """带本地缓存的数据获取"""
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


def find_first_limit_up(symbol, df):
    """识别连板的最后一个涨停板作为有效首板日"""
    market_type = "科创板" if symbol.startswith(("688", "689")) else "创业板" if symbol.startswith(
        ("300", "301")) else "主板"
    limit_rate = 0.20 if market_type in ["创业板", "科创板"] else 0.10
    # 计算涨停价
    df['prev_close'] = df['close'].shift(1)
    df['limit_price'] = (df['prev_close'] * (1 + limit_rate)).round(2)

    # 识别所有涨停日
    limit_days = df[df['close'] >= df['limit_price']].index.tolist()
    valid_days = []

    # 用于跟踪连板序列
    current_streak = []

    for i, day in enumerate(limit_days):
        # 检查是否是连板序列的开始或延续
        if not current_streak or (current_streak[-1] + pd.Timedelta(days=1) == day):
            current_streak.append(day)
        else:
            # 连板序列结束，取最后一个作为有效首板日（且要求连板长度至少为2）
            if len(current_streak) >= 2:  # 至少2连板
                last_limit_day = current_streak[-1]
                # 日期过滤条件
                if last_limit_day >= pd.Timestamp('2024-03-01'):
                    valid_days.append(last_limit_day)
            current_streak = [day]  # 开始新的连板序列

    # 处理最后一个连板序列
    if current_streak and len(current_streak) >= 2:  # 至少2连板
        last_limit_day = current_streak[-1]
        if last_limit_day >= pd.Timestamp('2024-03-01'):
            valid_days.append(last_limit_day)

    return valid_days

def generate_signals(df, first_limit_day, stock_code, stock_name):
    """生成买卖信号"""
    signals = []
    market_type = "科创板" if stock_code.startswith(("688", "689")) else "创业板" if stock_code.startswith(
        ("300", "301")) else "主板"
    limit_rate = 0.20 if market_type in ["创业板", "科创板"] else 0.10

    base_price = df.loc[first_limit_day, 'close'] # 首板收盘价，最重要的位置，表示主力的支撑度
    df['down_limit_price'] = (df['prev_close'] * (1 - limit_rate)).round(2)  # 新增跌停价字段
    min_price_threshold = base_price * 0.97  # 最低允许价格[6](@ref)

    start_idx = df.index.get_loc(first_limit_day)
    if (start_idx + 1) >= len(df):  # 新增边界检查，跳过无数据的情况
        return signals
    df['ma5'] = df['close'].rolling(5).mean()

    first_touch_flag = False

    for offset in range(2, 5):  # 最多检查20个交易日
        if start_idx + offset >= len(df):
            break

        current_day = df.index[start_idx + offset]
        current_data = df.iloc[start_idx + offset]

        # 获取最近5日MA5数据（防止空值）
        ma5_data = df['ma5'].iloc[start_idx:start_idx + offset + 1]
        if ma5_data.isnull().any():
            continue

        # 核心条件：当日最低价触碰五日均线
        touch_condition = (current_data['low'] <= current_data['ma5']) & \
                          (current_data['high'] >= current_data['ma5'])

        history_window = df.iloc[start_idx + 1: start_idx + offset]
        history_condition = (history_window['close'] > history_window['ma5']).all()

        if not first_touch_flag and touch_condition and history_condition:

            buy_price = current_data['ma5']  # 以五日均线值为买入价
            hold_days = 0

            # 卖出逻辑（保持原有逻辑）
            for sell_offset in range(1, 20):  # 最多持有20日
                if start_idx + offset + sell_offset >= len(df):
                    break

                sell_day = df.index[start_idx + offset + sell_offset]
                sell_data = df.loc[sell_day]
                hold_days += 1

                # 2. 断板日卖出（新增止盈条件）
                # 前一日涨停但当日未涨停（断板）
                if sell_data['close'] >= sell_data['limit_price']:
                    continue  # 涨停日继续持有

                prev_day = df.index[df.index.get_loc(sell_day) - 1]
                if df.loc[prev_day, 'close'] >= df.loc[prev_day, 'limit_price']:
                    profit_pct = (sell_data['close'] - buy_price) / buy_price * 100
                    signals.append({
                        '股票代码': stock_code,
                        '股票名称': stock_name,
                        '首板日': first_limit_day.strftime('%Y-%m-%d'),
                        '买入日': current_day.strftime('%Y-%m-%d'),
                        '卖出日': sell_day.strftime('%Y-%m-%d'),
                        '持有天数': hold_days,
                        '买入价': round(buy_price, 2),
                        '卖出价': round(sell_data['close'], 2),
                        '触碰类型': 'MA5支撑反弹' if current_data['close'] > current_data['ma5'] else 'MA5破位回升',
                        '收益率(%)': round(profit_pct, 2),
                        '卖出原因': '断板止盈'  # 新增字段
                    })
                    break

                # 3. 原有跌破五日线卖出条件
                if sell_data['close'] < sell_data['ma5']:
                    profit_pct = (sell_data['close'] - buy_price) / buy_price * 100
                    signals.append({
                        '股票代码': stock_code,
                        '股票名称': stock_name,
                        '首板日': first_limit_day.strftime('%Y-%m-%d'),
                        '买入日': current_day.strftime('%Y-%m-%d'),
                        '卖出日': sell_day.strftime('%Y-%m-%d'),
                        '持有天数': hold_days,
                        '买入价': round(buy_price, 2),
                        '卖出价': round(sell_data['close'], 2),
                        '触碰类型': 'MA5支撑反弹' if current_data['close'] > current_data['ma5'] else 'MA5破位回升',
                        '收益率(%)': round(profit_pct, 2),
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
                        '持有天数': hold_days,
                        '买入价': round(buy_price, 2),
                        '卖出价': round(sell_data['close'], 2),
                        '触碰类型': 'MA5支撑反弹' if current_data['close'] > current_data['ma5'] else 'MA5破位回升',
                        '收益率(%)': round(profit_pct, 2),
                        '卖出原因': '持有超限'
                    })
                    break
            break  # 只取第一次触碰
    return signals


def save_trades_excel(result_df):
    column_order = ['股票代码', '股票名称', '首板日', '买入日', '卖出日',
                    '持有天数', '买入价', '卖出价', '收益率(%)','卖出原因']
    result_df = result_df[column_order]
    """专业级Excel导出函数"""
    # 生成带时间戳的文件名[2,6](@ref)
    excel_name = f"首板交易记录_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

    # 创建带格式的Excel写入器[7](@ref)
    with pd.ExcelWriter(excel_name, engine='xlsxwriter',engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
        # 写入原始数据
        result_df.to_excel(writer, sheet_name='交易明细', index=False)

        # 获取工作表对象[6](@ref)
        workbook = writer.book
        worksheet = writer.sheets['交易明细']

        # ================== 专业格式设置 ==================
        # 1. 列宽自适应[2](@ref)
        for idx, col in enumerate(result_df.columns):
            series = result_df[col]
            max_len = max((
                series.astype(str).map(len).max(),  # 数据最大长度
                len(col)  # 列名长度
            )) + 2
            worksheet.set_column(idx, idx, max_len)

        # 2. 条件格式(收益率红涨绿跌)[7](@ref)
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

        # 3. 冻结首行[6](@ref)
        worksheet.freeze_panes(1, 0)

        # 4. 自动筛选[8](@ref)
        worksheet.autofilter(0, 0, len(result_df), len(result_df.columns) - 1)

        # 5. 添加统计页[5](@ref)
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

        # 6. 添加图表(示例)[4](@ref)
        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({
            'name': '收益率分布',
            'categories': ['交易明细', 1, 4, len(result_df), 4],  # 买入日列
            'values': ['交易明细', 1, 6, len(result_df), 6],  # 收益率列
            'data_labels': {'value': True, 'num_format': '0.00%'}
        })
        worksheet.insert_chart('J2', chart)

    print(f"\033[32m交易记录已保存至 {excel_name}\033[0m")


if __name__ == '__main__':
    # 获取当日涨停数据（新增）
    today = datetime.now()
    today_str = today.strftime("%Y%m%d")
    yesterday = today - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y%m%d")

    all_signals = []

    # 参数设置
    symbol = 'sh601086'  # 平安银行
    start_date = '20240201'

    query_tool = getAllStockCsv.StockQuery()
    # 加载股票列表并过滤
    filtered_stocks = query_tool.get_all_filter_stocks()
    stock_list = filtered_stocks[['stock_code', 'stock_name']].values
    total = len(stock_list)

    for idx, (code, name) in enumerate(stock_list, 1):
        df, _ = get_stock_data(code, start_date=start_date)
        if df.empty:
            continue

        first_limit_days = find_first_limit_up(code, df)
        for day in first_limit_days:
            signals = generate_signals(df, day, code, name)
            all_signals.extend(signals)

    # 生成统计报表[10](@ref)
    result_df = pd.DataFrame(all_signals)
    if not result_df.empty:
        win_rate = len(result_df[result_df['收益率(%)'] > 0]) / len(result_df) * 100
        avg_win = result_df[result_df['收益率(%)'] > 0]['收益率(%)'].mean()
        avg_loss = abs(result_df[result_df['收益率(%)'] <= 0]['收益率(%)'].mean())
        profit_ratio = avg_win / avg_loss if avg_loss != 0 else np.inf
        get_money=len(result_df[result_df['收益率(%)'] > 0]) / len(result_df)*avg_win-(1-len(result_df[result_df['收益率(%)'] > 0]) / len(result_df))*avg_loss

        print(f"\n\033[1m=== 策略表现汇总 ===\033[0m")
        print(f"总交易次数: {len(result_df)}")
        print(f"胜率: {win_rate:.1f}%")
        print(f"平均盈利: {avg_win:.1f}% | 平均亏损: {avg_loss:.1f}%")
        print(f"盈亏比: {profit_ratio:.2f}:1")
        print(f"期望收益: {get_money:.2f}")

        # 示例输出
        print("\n\033[1m最近5笔交易记录:\033[0m")
        print(result_df.tail(5).to_string(index=False))
    else:
        print("未产生有效交易信号")

    # 在生成result_df后调用
    if not result_df.empty:
        save_trades_excel(result_df)  # 新增调用
