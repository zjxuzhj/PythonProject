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
        if df.index.get_loc(day) >= 5:  # 确保有足够历史数据
            pre5_start = df.index[df.index.get_loc(day) - 5]
            pre5_close = df.loc[pre5_start, 'close']
            total_change = (df.loc[day, 'close'] - pre5_close) / pre5_close * 100
            if total_change >= 15:  # 累计涨幅≥5%则排除
                continue
        valid_days.append(day)
    return valid_days


def generate_signals(df, first_limit_day, stock_code, stock_name):
    """生成买卖信号（含分阶段止盈止损逻辑）"""
    signals = []
    market_type = "科创板" if stock_code.startswith(("688", "689")) else "创业板" if stock_code.startswith(
        ("300", "301")) else "主板"
    limit_rate = 0.20 if market_type in ["创业板", "科创板"] else 0.10

    base_price = df.loc[first_limit_day, 'close']
    df['down_limit_price'] = (df['prev_close'] * (1 - limit_rate)).round(2)

    start_idx = df.index.get_loc(first_limit_day)
    if (start_idx + 1) >= len(df):
        return signals

    # 初始化技术指标（网页2][网页5]）
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['vol_ma5'] = df['volume'].rolling(5).mean()

    first_touch_flag = False
    for offset in range(2, 20):  # 扩展观察期至20个交易日（网页3）
        if start_idx + offset >= len(df):
            break

        current_day = df.index[start_idx + offset]
        current_data = df.iloc[start_idx + offset]

        # 买入前一日收盘价 >= 首板涨停价
        # prev_day = df.index[start_idx + offset - 1]  # 买入前一日
        # if df.loc[prev_day, 'close'] < base_price:  # 跌破首板涨停价则跳过[4](@ref)
        #     continue

        # 有效性校验（网页1][网页4]）
        if check_invalid_condition(df, start_idx, offset):
            continue

        # 核心触发条件（网页5][网页7]）
        if (current_data['low'] <= current_data['ma5']) and \
                (current_data['high'] >= current_data['ma5']) and \
                (df.iloc[start_idx + 1:start_idx + offset]['close'] > df.iloc[start_idx + 1:start_idx + offset][
                    'ma5']).all():

            buy_price = round(current_data['ma5'], 2)
            position_size = 100  # 初始持仓100%
            half_sold = False
            exit_flag = False

            # 当触发6%卖出条件时
            sell_stages = []
            # 持仓监测循环（网页8]）
            for sell_offset in range(1, 11):  # 最长持有10交易日
                if start_idx + offset + sell_offset >= len(df):
                    break

                sell_day = df.index[start_idx + offset + sell_offset]
                sell_data = df.loc[sell_day]
                current_close = sell_data['close']
                current_high = sell_data['high']

                # ========== 阶段一：盘中超6%卖半仓 ==========
                if not half_sold and current_high >= buy_price * 1.06:
                    sell_stages.append({
                        'buy_price': buy_price,
                        'sell_day': sell_day,
                        'sell_price': current_high,
                        'position': 50,
                        'type': '6%止盈半仓'
                    })
                    half_sold = True

                # ========== 阶段二：剩余仓位处理 ==========
                exit_condition = [
                    current_close <= buy_price,  # 跌回买入价（网页4]
                    (current_close - buy_price) / buy_price >= 0.10,  # 收益超10%（网页2]
                    sell_offset >= 5  # 持有超5日（网页5]
                ]
                if any(exit_condition) and half_sold:
                    sell_stages.append({
                        'buy_price': buy_price,
                        'sell_day': sell_day,
                        'sell_price': current_close,
                        'position': 50,
                        'type': '剩余仓位退出'
                    })
                    signals.append(create_signal(
                        stock_code, stock_name, first_limit_day,
                        current_day, sell_stages
                    ))
                    break

                # ========== 未触发半仓卖出时的止损逻辑 ==========
                if not half_sold and current_close <= buy_price * 0.97:
                    signals.append(create_signal(
                        stock_code, stock_name, first_limit_day,
                        current_day, [{
                            'buy_price': buy_price,
                            'sell_day': sell_day,
                            'sell_price': current_close,
                            'position': 100,
                            'type': '3%止损'
                        }]
                    ))
                    break

                # 特殊处理：当日跌停强制清仓（网页1]
                if current_close <= sell_data['down_limit_price']:
                    signals.append(create_signal(
                        stock_code, stock_name, first_limit_day,
                        current_day, [{
                            'buy_price': buy_price,
                            'sell_day': sell_day,
                            'sell_price': current_close,
                            'position': 100,
                            'type': '跌停清仓'
                        }]
                    ))
                    exit_flag = True
                    break

            if exit_flag:
                break  # 终止外层循环
    return signals


def create_signal(code, name, limit_day, buy_day, sell_info):
    """支持多阶段操作的信号生成"""
    total_return = 0
    position_total = 0
    sell_dates = []
    operation_details = []

    hold_days = (sell_info[-1]['sell_day'] - buy_day).days

    # 遍历所有卖出阶段
    for stage in sell_info:
        partial_return = (stage['sell_price'] / stage['buy_price'] - 1) * (stage['position'] / 100)
        total_return += partial_return
        position_total += stage['position']
        sell_dates.append(stage['sell_day'].strftime('%m-%d'))
        operation_details.append(
            f"{stage['position']}%@{stage['sell_price']}({stage['type']})"
        )

    # 仓位校验
    if position_total != 100:
        raise ValueError("仓位比例总和必须为100%")

    return {
        '股票代码': code,
        '股票名称': name,
        '首板日': limit_day.strftime('%Y-%m-%d'),
        '买入日': buy_day.strftime('%Y-%m-%d'),
        '卖出阶段': "→".join(sell_dates),
        '买入价': sell_info[0]['buy_price'],  # 取首次买入价
        '卖出均价': round(
            sum(s['sell_price'] * s['position'] for s in sell_info) / 100, 2
        ),
        '持有天数': hold_days,
        '仓位比例(%)': 100,
        '操作明细': " + ".join(operation_details),
        '总收益率(%)': round(total_return * 100, 2),
        '交易ID': f"{code}_{buy_day.strftime('%Y%m%d')}"
    }


def check_invalid_condition(df, start_idx, offset):
    """校验无效交易条件（网页1][网页4]）"""
    # 前五日跌停检测
    check_period = df.iloc[max(0, start_idx + offset - 5):start_idx + offset]
    if (check_period['close'] <= check_period['down_limit_price']).any():
        return True

    # 前六日涨停次数检测
    limit_check = df.iloc[max(0, start_idx + offset - 6):start_idx + offset]
    if (limit_check['close'] >= limit_check['limit_price']).sum() > 1:
        return True

    # 量能异常检测（网页7]）
    if df.iloc[start_idx + offset]['volume'] < df.iloc[start_idx + offset]['vol_ma5'] * 0.5:
        return True

    return False


def create_signal(code, name, limit_day, buy_day, sell_day, hold_days,
                  buy_price, sell_price, position, signal_type):
    """标准化信号生成"""
    return {
        '股票代码': code,
        '股票名称': name,
        '首板日': limit_day.strftime('%Y-%m-%d'),
        '买入日': buy_day.strftime('%Y-%m-%d'),
        '卖出日': sell_day.strftime('%Y-%m-%d'),
        '持有天数': hold_days,
        '买入价': buy_price,
        '卖出价': round(sell_price, 2),
        '仓位比例(%)': position,
        '操作类型': signal_type,
        '收益率(%)': round((sell_price / buy_price - 1) * (position / 100) * 100, 2),
        '交易ID': f"{code}_{buy_day}",  # 增加唯一标识
    }


def check_invalid_condition(df, start_idx, offset):
    """校验无效交易条件"""
    # 前五日跌停检测
    check_period = df.iloc[max(0, start_idx + offset - 5):start_idx + offset]
    if (check_period['close'] <= check_period['down_limit_price']).any():
        return True

    # 前六日涨停次数检测
    limit_check = df.iloc[max(0, start_idx + offset - 6):start_idx + offset]
    if (limit_check['close'] >= limit_check['limit_price']).sum() > 1:
        return True

    # 量能异常检测
    if df.iloc[start_idx + offset]['volume'] < df.iloc[start_idx + offset]['vol_ma5'] * 0.5:
        return True

    return False

def save_trades_excel(result_df):
    column_order = [
        '股票代码', '股票名称', '首板日', '买入日',
        '卖出阶段', '买入价', '卖出均价',
        '总收益率(%)', '操作明细'
    ]
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

    # 合并交易记录
    merged_signals = []
    seen_ids = set()

    # 生成统计报表[10](@ref)
    for sig in all_signals:
        if sig['交易ID'] not in seen_ids:
            merged_signals.append(sig)
            seen_ids.add(sig['交易ID'])

    result_df = pd.DataFrame(merged_signals)

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
