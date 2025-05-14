import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import getAllStockCsv


def get_stock_data(symbol, start_date, force_update=False):
    """带本地缓存的数据获取"""
    # 生成唯一文件名（网页1）
    file_name = f"stock_{symbol}_{start_date}.parquet"
    cache_path = os.path.join("data_cache", file_name)

    # 非强制更新时尝试读取缓存
    if not force_update and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path, engine='fastparquet')
            # df['vol_ma5'] = df['volume'].rolling(5).mean()
            # df['volume_ratio'] = df['volume'] / df['vol_ma5'].replace(0, 1)  # 防除零错误
            print(f"从缓存加载数据：{symbol}")
            return df, True  # 返回缓存标记
        except Exception as e:
            print(f"缓存读取失败：{e}（建议删除损坏文件：{cache_path}）")

    # 强制更新或缓存不存在时获取新数据（网页7）
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
        # 排除连板(次日不涨停)
        next_day = df.index[df.index.get_loc(day) + 1] if (df.index.get_loc(day) + 1) < len(df) else None
        if next_day and df.loc[next_day, 'close'] >= df.loc[next_day, 'limit_price']:
            continue

        # 新增日期过滤条件（网页1]）
        if day < pd.Timestamp('2024-03-01'):
            continue

        # # 新增：前五日累计涨幅校验
        # if df.index.get_loc(day) >= 5:  # 确保有足够历史数据
        #     pre5_start = df.index[df.index.get_loc(day) - 5]
        #     pre5_close = df.loc[pre5_start, 'close']
        #     total_change = (df.loc[day, 'close'] - pre5_close) / pre5_close * 100
        #     if total_change <= 15:  # 累计涨幅≥5%则排除
        #         continue
        valid_days.append(day)
    return valid_days


def generate_signals(df, first_limit_day, stock_code, stock_name):
    """生成买卖信号（修改版）"""
    signals = []
    start_idx = df.index.get_loc(first_limit_day)

    # 获取涨停日数据[3](@ref)
    limit_day_data = df.iloc[start_idx]
    limit_volume = limit_day_data['volume']

    # 只处理涨停次日[3](@ref)
    if start_idx + 1 >= len(df):
        return signals
    buy_day = df.index[start_idx + 1]
    buy_data = df.loc[buy_day]

    # 条件1：成交量≥涨停日2倍
    if buy_data['volume'] < limit_volume * 2:
        return signals

    # 条件2：小阳线（涨幅2%-4.5%）
    price_change = (buy_data['close'] - buy_data['prev_close']) / buy_data['prev_close'] * 100
    if not (1.5 <= price_change <= 4.5):
        return signals

    # 条件3：光头或短上影线（上影线≤实体长度1/10）
    body = buy_data['close'] - buy_data['open']
    upper_shadow = buy_data['high'] - buy_data['close']
    if upper_shadow > abs(body) * 0.15:
        return signals

    # 计算不同持有期收益率[15,16](@ref)
    holding_periods = [1, 3, 5, 10]
    returns = {}

    for period in holding_periods:
        if start_idx + 1 + period >= len(df):
            returns[f'{period}天收益率(%)'] = None
            continue

        sell_day = df.index[start_idx + 1 + period]
        sell_price = df.loc[sell_day, 'close']
        returns[f'{period}天收益率(%)'] = round(
            (sell_price - buy_data['close']) / buy_data['close'] * 100, 2)

    signals.append({
        '股票代码': stock_code,
        '股票名称': stock_name,
        '首板日': first_limit_day.strftime('%Y-%m-%d'),
        '买入日': buy_day.strftime('%Y-%m-%d'),
        '买入价': round(buy_data['close'], 2),
        **returns  # 展开不同持有期收益
    })
    return signals


def save_trades_excel(result_df):
    """专业级Excel导出函数（多周期版本）"""
    # 调整列顺序[6](@ref)
    column_order = [
        '股票代码', '股票名称', '首板日', '买入日', '买入价',
        '1天收益率(%)', '3天收益率(%)', '5天收益率(%)', '10天收益率(%)'
    ]
    result_df = result_df[column_order]

    excel_name = f"首板交易记录_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

    with pd.ExcelWriter(excel_name, engine='xlsxwriter',
                        engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
        result_df.to_excel(writer, sheet_name='交易明细', index=False)

        workbook = writer.book
        worksheet = writer.sheets['交易明细']

        # 设置自适应列宽[2](@ref)
        for idx, col in enumerate(result_df.columns):
            series = result_df[col].astype(str)
            max_len = max(series.str.len().max(), len(col)) + 2
            worksheet.set_column(idx, idx, max_len * 1.2)

        # 条件格式设置[6](@ref)
        format_green = workbook.add_format({'font_color': '#00B050', 'num_format': '0.00%'})
        format_red = workbook.add_format({'font_color': '#FF0000', 'num_format': '0.00%'})

        # 为每个收益率列设置格式[7](@ref)
        return_cols = [col for col in result_df.columns if '收益率' in col]
        for col in return_cols:
            col_idx = result_df.columns.get_loc(col)
            for row in range(1, len(result_df) + 1):
                cell_value = result_df.iloc[row - 1][col]
                if pd.isna(cell_value):
                    continue
                cell_value /= 100
                if cell_value >= 0:
                    worksheet.write(row, col_idx, cell_value, format_green)
                else:
                    worksheet.write(row, col_idx, cell_value, format_red)

        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, len(result_df), len(result_df.columns) - 1)

        # 多周期统计页[5](@ref)
        stats_data = []
        periods = [1, 3, 5, 10]
        for period in periods:
            col = f'{period}天收益率(%)'
            valid_data = result_df[col].dropna()

            stats_data.append({
                '持有周期': f'{period}天',
                '有效样本数': len(valid_data),
                '胜率(%)': valid_data.gt(0).mean() * 100,
                '平均收益(%)': valid_data.mean(),
                '最大收益(%)': valid_data.max(),
                '最大亏损(%)': valid_data.min()
            })

        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='策略统计', index=False)

        # 添加组合图表[4](@ref)
        chart = workbook.add_chart({'type': 'line'})
        for i, period in enumerate(periods):
            col_name = f'{period}天收益率(%)'
            col_idx = result_df.columns.get_loc(col_name)
            chart.add_series({
                'name': col_name,
                'categories': ['交易明细', 1, 3, len(result_df), 3],  # 买入日列
                'values': ['交易明细', 1, col_idx, len(result_df), col_idx],
                'line': {'color': ['#FF9900', '#3366CC', '#109618', '#990099'][i]}
            })
        chart.set_title({'name': '多周期收益趋势对比'})
        worksheet.insert_chart('J20', chart)

    print(f"\033[32m交易记录已保存至 {excel_name}\033[0m")


if __name__ == '__main__':
    today = datetime.now()
    start_date = '20240201'

    query_tool = getAllStockCsv.StockQuery()
    filtered_stocks = query_tool.get_all_filter_stocks()
    stock_list = filtered_stocks[['stock_code', 'stock_name']].values

    all_signals = []
    for idx, (code, name) in enumerate(stock_list, 1):
        df, _ = get_stock_data(code, start_date=start_date)
        if df.empty:
            continue

        first_limit_days = find_first_limit_up(code, df)
        for day in first_limit_days:
            signals = generate_signals(df, day, code, name)
            all_signals.extend(signals)

    result_df = pd.DataFrame(all_signals)

    if not result_df.empty:
        # 多周期统计输出[6](@ref)
        print("\n\033[1m=== 策略表现汇总 ===\033[0m")
        print(f"总交易次数: {len(result_df)}")

        for period in [1, 3, 5, 10]:
            col = f'{period}天收益率(%)'
            valid_data = result_df[col].dropna()
            if len(valid_data) == 0:
                continue

            win_rate = valid_data.gt(0).mean() * 100
            avg_return = valid_data.mean()
            print(f"\n--- {period}天收益率统计 ---")
            print(f"有效样本数: {len(valid_data)}")
            print(f"胜率: {win_rate:.1f}%")
            print(f"平均收益: {avg_return:.1f}%")

        # 示例输出
        print("\n\033[1m最近5笔交易记录:\033[0m")
        print(result_df.tail(5).to_string(index=False))

        save_trades_excel(result_df)
    else:
        print("未产生有效交易信号")
