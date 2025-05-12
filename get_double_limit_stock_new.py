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
    day1 = df.index[start_idx + 1]
    if df.loc[day1, 'close'] < base_price: # 首板次日低于首板收盘价就跳过
        return signals

    df['ma55'] = df['close'].rolling(60).mean()
    df['ma30'] = df['close'].rolling(30).mean()
    df['ma5'] = df['close'].rolling(5).mean()

    for offset in range(2, 6):  # 首板后第2-6日(共4日，不包括6)
        if start_idx + offset >= len(df):
            break
        current_day = df.index[start_idx + offset]
        # 检查首板次日至买入前一日的收盘价
        price_valid = True
        for check_day in range(start_idx + 1, start_idx + offset):  # 遍历中间交易日
            current_close = df.iloc[check_day]['close']
            if current_close < base_price:
                price_valid = False
                break
            # if current_close > base_price * 1.04:
            #     price_valid = False
            #     print(f"排除：{stock_code} 在 {df.index[check_day].strftime('%Y-%m-%d')} 收盘价超过首板价4%")
            #     break

        if not price_valid:
            continue  # 存在跌破阈值日则跳过

        # # ================ 新增条件1：前5日无跌停 ================
        # # 获取当前日的前5个交易日范围
        # start_check_idx = max(0, start_idx + offset - 5)
        # check_period = df.iloc[start_check_idx: start_idx + offset]
        # # 检查是否存在跌停(收盘价<=跌停价)
        # has_down_limit = (check_period['close'] <= check_period['down_limit_price']).any()
        # if has_down_limit:
        #     continue
        #
        # # ================ 新增条件2：前6日涨停次数≤1 ================
        # # 获取当前日的前6个交易日范围
        # start_limit_check = max(0, start_idx + offset - 6)
        # limit_check_period = df.iloc[start_limit_check: start_idx + offset]
        # # 统计涨停次数(排除首板日自身)
        # limit_count = (limit_check_period['close'] >= limit_check_period['limit_price']).sum()
        # if limit_count > 1:  # 包含当天则为>1，不包含则为>=1
        #     continue

        # 新增55日线压制校验(网页6)
        # ma55_valid = True
        # for check_day in range(start_idx + 1, start_idx + offset):
        #     current_close = df.iloc[check_day]['close']
        #     current_ma55 = df.iloc[check_day]['ma55']
        #     if current_close < current_ma55:
        #         ma55_valid = False
        #         break
        #
        # if not ma55_valid:
        #     continue  # 存在压制日则跳过

        current_data = df.loc[current_day]

        # # 优化55日线压制校验（符合多次假突破特征）[6,7](@ref)
        # ma55_valid = True
        # break_count = 0  # 突破失败计数器
        #
        # for check_day in range(start_idx + 1, start_idx + offset):
        #     current_high = df.iloc[check_day]['high']
        #     current_close = df.iloc[check_day]['close']
        #     current_ma55 = df.iloc[check_day]['ma55']
        #
        #     # 满足：当日最高价突破MA55 且 收盘价回落
        #     if (current_high > current_ma55) and (current_close < current_ma55):
        #         break_count += 1
        #
        #     # 提前终止条件：当出现两次即确认压制
        #     if break_count > 1:
        #         ma55_valid = False
        #         break
        #
        # if not ma55_valid:
        #     continue  # 存在有效压制则跳过

        # 新增：首板后每日不破5日线校验（网页1][网页4]）
        # ma5_valid = True
        # for check_day in range(start_idx + 1, start_idx + offset):
        #     current_close = df.iloc[check_day]['close']
        #     current_ma5 = df.iloc[check_day]['ma5']
        #
        #     # 当日收盘价必须≥5日均线（网页2][网页5]）
        #     if current_close < current_ma5:
        #         ma5_valid = False
        #         break
        #
        # if not ma5_valid:
        #     continue  # 存在破位日则跳过


        # 买入条件：盘中破首板价但收盘收复[3](@ref)
        if all([
            # current_data['low'] < base_price,  # 盘中破首板价
            # current_data['close'] >= base_price,  # 收盘收复
            # current_data['close'] < current_data['limit_price'],  # 非涨停收盘
            # (current_data['close'] - current_data['prev_close']) / current_data['prev_close'] <= 0.05,  # 当日涨幅≤5%
            # current_data['volume_ratio'] < 0.7,
            price_valid,  # 新增价格校验
            # ma5_valid
        ]):
            buy_price = current_data['close']
            sell_info = None
            hold_days = 0

            # 卖出条件监测
            for sell_offset in range(1, 6):  # 最多持有5日
                if start_idx + offset + sell_offset >= len(df):
                    break
                sell_day = df.index[start_idx + offset + sell_offset]
                sell_data = df.loc[sell_day]
                hold_days += 1

                # 触发卖出条件
                if any([
                    sell_data['close'] < buy_price * 0.97,  # 原基于买入价的止损跌破3%
                    # sell_data['close'] < base_price * 0.97,  # 新基于首板收盘价的止损
                    sell_data['close'] >= sell_data['limit_price'],  # 涨停卖出
                    (sell_data['close'] - buy_price) / buy_price >= 0.15,  # 盈利10%
                    hold_days == 5  # 最大持有
                ]):
                    sell_price = sell_data['close']
                    profit_pct = (sell_price - buy_price) / buy_price * 100
                    signals.append({
                        '股票代码': stock_code,
                        '股票名称': stock_name,
                        '首板日': first_limit_day.strftime('%Y-%m-%d'),
                        '买入日': current_day.strftime('%Y-%m-%d'),
                        '卖出日': sell_day.strftime('%Y-%m-%d'),
                        '持有天数': hold_days,
                        '买入价': round(buy_price, 2),
                        '卖出价': round(sell_price, 2),
                        '收益率(%)': round(profit_pct, 2)
                    })
                    break
            break  # 只做第一次符合条件的买入
    return signals


def save_trades_excel(result_df):
    column_order = ['股票代码', '股票名称', '首板日', '买入日', '卖出日',
                    '持有天数', '买入价', '卖出价', '收益率(%)']
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
