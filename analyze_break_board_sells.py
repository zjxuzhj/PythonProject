import os
import time
from datetime import datetime, timedelta
import pandas as pd
from xtquant import xtdata

# --- 1. 配置区域 ---

# 请将 '首板交易记录_YYYYMMDD_HHMM.xlsx' 替换为您实际的回测结果文件名
# 这个文件应该是由您之前的 first_limit_up_ma5_normal.py 脚本生成的
RESULTS_FILE_PATH = '首板交易记录_20250808_1645.xlsx'

# 分析结果将保存到这个CSV文件中
OUTPUT_CSV_PATH = '断板止盈策略分析结果.csv'


# --- 2. 核心功能函数 ---

def load_backtest_results(file_path: str) -> pd.DataFrame:
    """从Excel文件加载回测结果，并筛选出'断板止盈'的交易。"""
    if not os.path.exists(file_path):
        print(f"错误：回测结果文件未找到 -> {file_path}")
        return pd.DataFrame()

    print(f"正在从 {file_path} 加载数据...")
    df = pd.read_excel(file_path)

    # 筛选出卖出原因是“断板止盈”的记录
    filtered_df = df[df['卖出原因'] == '断板止盈'].copy()
    print(f"成功加载并筛选出 {len(filtered_df)} 条 '断板止盈' 的交易记录。")
    return filtered_df


def get_1min_data(stock_code: str, date_str: str) -> pd.DataFrame:
    """使用xtdata获取指定股票和日期的1分钟K线数据。"""
    try:
        # 将 'sh600519' 格式转换为 '600519.SH'
        qmt_code = f"{stock_code[2:]}.{stock_code[:2].upper()}"

        # 获取数据
        df_dict = xtdata.get_local_data(
            stock_list=[qmt_code],
            period='1m',
            start_time=date_str,
            end_time=date_str
        )

        if qmt_code in df_dict and not df_dict[qmt_code].empty:
            df = df_dict[qmt_code]
            df['datetime'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
            df.dropna(subset=['datetime'], inplace=True)
            if df.empty:
                print("警告：当天没有有效的时间数据。")
            else:
                df.set_index('datetime', inplace=True)
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Shanghai')
            return df
        else:
            print(f"警告：未获取到 {stock_code} 在 {date_str} 的1分钟数据。")
            return pd.DataFrame()

    except Exception as e:
        print(f"错误：获取 {stock_code} 在 {date_str} 的1分钟数据时发生异常: {e}")
        return pd.DataFrame()


def get_previous_day_close(stock_code: str, date_str: str) -> float:
    """获取指定日期前一个交易日的收盘价。"""
    try:
        qmt_code = f"{stock_code[2:]}.{stock_code[:2].upper()}"

        # 为了获取前一天的收盘价，我们需要请求包含前一天在内的数据
        start_date = (datetime.strptime(date_str, '%Y%m%d') - timedelta(days=5)).strftime('%Y%m%d')

        df_dict = xtdata.get_local_data(
            stock_list=[qmt_code],
            period='1d',
            start_time=start_date,
            end_time=date_str
        )

        if qmt_code in df_dict and len(df_dict[qmt_code]) >= 2:
            # -2 定位到倒数第二行，即前一天的收盘价
            return df_dict[qmt_code]['close'].iloc[-2]
        else:
            return 0.0

    except Exception:
        return 0.0


def calculate_vwap(df_subset: pd.DataFrame) -> float:
    """为1分钟数据子集计算成交量加权平均价(VWAP)。"""
    if df_subset.empty:
        return 0.0

    # VWAP = Σ(价格 * 成交量) / Σ(成交量)
    # 这里我们用每分钟的收盘价作为该分钟的平均价格来估算
    total_value = (df_subset['close'] * df_subset['volume']).sum()
    total_volume = df_subset['volume'].sum()

    if total_volume == 0:
        return 0.0

    return total_value / total_volume


# --- 3. 主逻辑 ---

def main():
    """主执行函数"""
    print("--- 开始分析'断板止盈'卖出策略 ---")

    # 连接到MiniQMT
    xtdata.connect()

    # 加载并筛选数据
    trades_to_analyze = load_backtest_results(RESULTS_FILE_PATH)
    if trades_to_analyze.empty:
        xtdata.disconnect()
        return

    analysis_results = []
    total_trades = len(trades_to_analyze)

    # 遍历每一笔符合条件的交易
    for idx, row in trades_to_analyze.iterrows():
        stock_code = row['股票代码']
        sell_date_str = pd.to_datetime(row['卖出日']).strftime('%Y%m%d')
        original_eod_price = row['卖出价']  # 这是原始策略在尾盘卖出的价格

        print(f"\n({idx + 1}/{total_trades}) 正在分析: {stock_code} 在 {sell_date_str} 的交易...")

        # 获取当天的1分钟数据
        min_df = get_1min_data(stock_code, sell_date_str)
        if min_df.empty:
            continue

        # 检查是否低开
        day_open_price = min_df['open'].iloc[0]
        prev_day_close = get_previous_day_close(stock_code, sell_date_str)
        is_low_open = day_open_price < prev_day_close if prev_day_close > 0 else False

        # 计算低开幅度
        low_open_pct = 0.0
        if prev_day_close > 0:
            low_open_pct = (day_open_price - prev_day_close) / prev_day_close * 100

        print(
            f"前日收盘: {prev_day_close:.2f}, 今日开盘: {day_open_price:.2f} -> 是否低开: {is_low_open}, 幅度: {low_open_pct:.2f}%")

        # 定位10:30的数据（注意：您代码中已改为10:30）
        target_time = pd.Timestamp(f"{sell_date_str} 10:30:00", tz='Asia/Shanghai')
        if target_time in min_df.index:
            bar_10am = min_df.loc[target_time]
            price_at_10am = bar_10am['close']

            # 计算截至10:30的VWAP
            df_until_10am = min_df.loc[min_df.index <= target_time]
            vwap_at_10am = calculate_vwap(df_until_10am)

            print(f"10:30 时价格: {price_at_10am:.2f}, 10:30 时均价(VWAP): {vwap_at_10am:.2f}")

            # 判断新策略是否会触发卖出
            should_sell_at_10am = is_low_open and (price_at_10am < vwap_at_10am)

            # 获取尾盘14:59的价格作为对比
            eod_time = pd.Timestamp(f"{sell_date_str} 14:59:00", tz='Asia/Shanghai')  # 同样加上时区
            if eod_time in min_df.index:
                price_at_eod = min_df.loc[eod_time]['close']
            else:
                price_at_eod = original_eod_price  # 如果没有14:59数据，则用原始数据

            # 记录分析结果
            analysis_results.append({
                '股票代码': stock_code,
                '股票名称': row['股票名称'],
                '卖出日': row['卖出日'],
                '是否低开': is_low_open,
                '低开幅度(%)': low_open_pct,
                '10点30价格': price_at_10am,  # 修改标签以匹配时间
                '10点30均价': vwap_at_10am,  # 修改标签以匹配时间
                '是否触发10点30卖出': should_sell_at_10am,  # 修改标签以匹配时间
                '10点30卖出价': price_at_10am if should_sell_at_10am else None,  # 修改标签以匹配时间
                '尾盘卖出价': price_at_eod,
                '原始策略卖出价': original_eod_price
            })
        else:
            print("警告: 当天10:30没有1分钟K线数据（可能停牌或一字板）。")

    # --- 4. 结果处理和保存 ---

    if not analysis_results:
        print("\n没有可供分析的数据。")
        xtdata.disconnect()
        return

    # 创建结果DataFrame
    analysis_df = pd.DataFrame(analysis_results)

    # 计算价格差异
    def calculate_diff(row):
        if row['是否触发10点30卖出']:
            # 使用 .get() 避免因列名修改而报错
            sell_price_1030 = row.get('10点30卖出价')
            eod_price = row.get('尾盘卖出价')
            if sell_price_1030 is not None and eod_price is not None and eod_price > 0:
                return (sell_price_1030 - eod_price) / eod_price * 100
        return None

    analysis_df['10点30卖出vs尾盘收益差(%)'] = analysis_df.apply(calculate_diff, axis=1)

    # 格式化输出
    format_cols = ['10点30价格', '10点30均价', '10点30卖出价', '尾盘卖出价', '原始策略卖出价']
    for col in format_cols:
        if col in analysis_df.columns:
            analysis_df[col] = analysis_df[col].map(lambda x: f"{x:.2f}" if pd.notnull(x) else '')

    analysis_df['低开幅度(%)'] = analysis_df['低开幅度(%)'].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else '')
    analysis_df['10点30卖出vs尾盘收益差(%)'] = analysis_df['10点30卖出vs尾盘收益差(%)'].map(
        lambda x: f"{x:.2f}%" if pd.notnull(x) else '')

    # 保存到CSV
    analysis_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

    print(f"\n--- 分析完成 ---")
    print("最终分析结果:")
    print(analysis_df.to_string())
    print(f"\n详细结果已保存至: {OUTPUT_CSV_PATH}")

    # 断开连接
    xtdata.disconnect()


if __name__ == '__main__':
    main()