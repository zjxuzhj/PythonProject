import pandas as pd
import numpy as np
import argparse
import datetime
import sys
import os

# 策略参数
PLATFORM_DAYS_DEFAULT = 10
PLATFORM_DAYS_LONG = 20
VOLUME_SURGE_RATIO = 1.5
VOLUME_SHRINK_RATIO = 1.0

# 默认配置（如果没有提供命令行参数）
DEFAULT_STOCK_CODE = "601179"  # 默认股票代码 (例如: 中百)
DEFAULT_MENTION_DATE = "2026-02-03"  # 默认首次提及日期

# 模拟策略配置类，用于兼容提供的get_stock_data逻辑
class StrategyConfig:
    def __init__(self):
        self.USE_2019_DATA = False # 默认使用20240201的数据

def get_stock_data(symbol, config=None):
    """带本地缓存的数据获取"""
    if config is None:
        config = StrategyConfig()
        
    date_str = "20190101" if config.USE_2019_DATA else "20240201"
    cache_dir = "back_test_data_cache" if config.USE_2019_DATA else "data_cache"
    # 获取项目根目录（当前文件在stock_analysis目录下，根目录是上一级）
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 尝试多种代码格式：纯数字、带sh/sz前缀
    possible_filenames = []
    
    # 如果输入是纯数字，尝试加前缀
    if symbol.isdigit():
        prefix = "sz" if symbol.startswith(("0", "3")) else "sh"
        if symbol.startswith(("8", "4", "9")): # 北交所
             prefix = "bj"
        possible_filenames.append(f"stock_{prefix}{symbol}_{date_str}.parquet")
    else:
        # 已经带前缀
        possible_filenames.append(f"stock_{symbol}_{date_str}.parquet")
    
    df = pd.DataFrame()
    found = False
    
    for filename in possible_filenames:
        file_path = os.path.join(root_dir, cache_dir, filename)
        if os.path.exists(file_path):
            try:
                # 尝试加载parquet，如果失败则提示
                df = pd.read_parquet(file_path, engine='fastparquet')
                print(f"已从缓存加载数据: {filename}")
                found = True
                break
            except Exception as e:
                print(f"读取缓存文件失败 {file_path}: {e}")
                return None
        else:
            pass
            
    if not found:
        print(f"数据文件未找到: {symbol} (尝试路径: {os.path.join(root_dir, cache_dir)})")
        return None

    if df.empty:
        return None
        
    # 重命名列以匹配策略使用的英文列名
    # parquet文件列名通常是：date, open, close, high, low, volume, amount...
    # 或者中文列名。这里做个映射检查。
    
    # 假设parquet列名可能为中文，尝试重命名
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    }
    df.rename(columns=rename_map, inplace=True)
    
    # 确保date列是datetime并设为索引
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    df.sort_index(inplace=True)
    
    return df

def calculate_indicators(df):
    """计算策略所需的技术指标"""
    # 移动平均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()
    
    # 成交量均线
    df['MA_Vol_5'] = df['volume'].rolling(window=5).mean()
    
    # 重心: (最高 + 最低 + 收盘) / 3
    df['center'] = (df['high'] + df['low'] + df['close']) / 3
    
    # 昨日数据 (shift)
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_center'] = df['center'].shift(1)
    df['prev_MA5'] = df['MA5'].shift(1)
    
    return df

def check_strategy(stock_code, first_mention_date_str=None, platform_days=PLATFORM_DAYS_DEFAULT):
    print(f"\n分析股票策略: {stock_code}")
    if first_mention_date_str:
        print(f"焦佬首次提及日期: {first_mention_date_str}")
    
    df = get_stock_data(stock_code)
    if df is None:
        return

    df = calculate_indicators(df)
    
    # 如果提供了首次提及日期，转换为datetime
    first_mention_dt = None
    if first_mention_date_str:
        try:
            # 尝试解析多种格式
            first_mention_dt = pd.to_datetime(first_mention_date_str)
        except:
            print(f"警告: 无法解析日期 {first_mention_date_str}。已忽略。")

    # 如果提供了提及日期，从该日期开始切片分析，否则分析近期数据
    analysis_df = df
    if first_mention_dt:
        analysis_df = df[df.index >= first_mention_dt]
        if analysis_df.empty:
            print("提及日期之后无可用数据。")
            return

    # 模拟策略执行
    # 我们将遍历每一天以寻找提及后的第一个买点
    # 然后跟踪状态
    
    position = 0 # 0: 空仓, 1: 底仓 (20%), 2: 加仓 (50%), 3: 重仓 (80%)
    holdings_pct = 0
    entry_price = 0
    entry_date = None
    logs = []

    # 遍历分析周期
    for date, row in analysis_df.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        
        # --- 第一阶段: 观察与建仓 (0 -> 1) ---
        if position == 0:
            # 红线检查: 下跌趋势 (MA5 < MA10 < MA20) -> 跳过
            if row['MA5'] < row['MA10'] and row['MA10'] < row['MA20']:
                logs.append(f"[{date_str}] 跳过: 下跌趋势 (MA5<MA10<MA20)。")
                continue
                
            # 入场条件 (满足 A, B, C 任一)
            buy_signal = False
            reason = ""
            
            # 条件 A: 均线确认
            # 突破并站稳 MA5, MA5 走平或向上
            ma5_slope_up = row['MA5'] >= row['prev_MA5']
            cond_a = (row['close'] > row['MA5']) and ma5_slope_up
            
            # 条件 B: 形态确认 (底分型)
            # 收盘 > 昨日最高 (反包) 且 重心 > 昨日重心
            cond_b = (row['close'] > row['prev_high']) and (row['center'] > row['prev_center'])
            
            # 条件 C: 支撑确认
            # 回踩 MA60 但收盘在上方, 缩量
            # "回踩" 意味着 最低价 <= MA60 * 1.01 (接近)
            cond_c_touch = (row['low'] <= row['MA60'] * 1.01) and (row['close'] > row['MA60'])
            cond_c_vol = row['volume'] < row['MA_Vol_5'] * VOLUME_SHRINK_RATIO
            cond_c = cond_c_touch and cond_c_vol
            
            if cond_a:
                buy_signal = True
                reason += "条件 A (站稳 MA5 且 向上); "
            if cond_b:
                buy_signal = True
                reason += "条件 B (底分型: 收盘>昨日最高 且 重心上移); "
            if cond_c:
                buy_signal = True
                reason += "条件 C (MA60 支撑 且 缩量); "
                
            if buy_signal:
                position = 1
                holdings_pct = 20
                entry_price = row['close']
                entry_date = date
                logs.append(f"[{date_str}] 买入底仓 (20%): {reason} 价格: {row['close']:.2f}")
                
        # --- 第二阶段: 加仓 (1 -> 2 -> 3) ---
        elif position >= 1:
            # 加仓前提: 盈利
            # 计算当前盈亏比例
            profit_pct = (row['close'] - entry_price) / entry_price * 100
            is_profitable = profit_pct > 0
            
            # 先进行 卖出/止损 逻辑检查
            # 止损 / 趋势破坏
            # 规则: 有效跌破 MA10 -> 清仓
            if row['close'] < row['MA10']:
                position = 0
                holdings_pct = 0
                logs.append(f"[{date_str}] 全部卖出 (清仓): 跌破 MA10。 价格: {row['close']:.2f} 盈亏: {profit_pct:.2f}%")
                entry_price = 0
                entry_date = None
                continue
            
            # 规则: 跌破 MA5 但守住 MA10 -> 减仓
            if row['close'] < row['MA5'] and row['close'] > row['MA10']:
                 # 如果是重仓，减仓。如果是底仓，可能持有或减仓。
                 # 策略说 "减仓 30%-50%"。
                 if position > 1:
                     position -= 1 # 简单的降级逻辑
                     holdings_pct = max(20, holdings_pct - 30)
                     logs.append(f"[{date_str}] 减仓: 跌破 MA5 但守住 MA10。 新仓位: {holdings_pct}%")
            
            # 加仓逻辑
            if is_profitable and position < 3: # 最多 3 层 (约 80%)
                add_signal = False
                add_reason = ""
                
                # 场景 A: 平台突破 (暴力)
                # 回溯 N 天寻找高点
                # 我们需要这一行之前的历史数据来检查平台
                # 获取窗口 [date - N days : date - 1 day]
                # 高效方法: 切片 df
                idx = df.index.get_loc(date)
                if idx > platform_days:
                    # 从昨天往前推 'platform_days' 天
                    past_window = df.iloc[idx-platform_days : idx]
                    platform_high = past_window['high'].max()
                    
                    cond_break = row['close'] > platform_high
                    cond_vol = row['volume'] > row['MA_Vol_5'] * VOLUME_SURGE_RATIO
                    
                    if cond_break and cond_vol:
                        add_signal = True
                        add_reason += f"场景 A (平台突破 > {platform_high:.2f} 且 放量); "
                
                # 场景 B: 趋势回踩 (温和)
                # 触碰 MA5/10, 收盘 > MA10, 缩量, 红K线/假阴线
                cond_touch = (row['low'] <= row['MA5']) or (row['low'] <= row['MA10'])
                cond_hold = row['close'] > row['MA10']
                cond_vol_shrink = row['volume'] < row['MA_Vol_5'] * VOLUME_SHRINK_RATIO
                
                is_red = row['close'] > row['open']
                is_fake_yin = (row['close'] < row['open']) and (row['close'] > row['prev_close'])
                cond_candle = is_red or is_fake_yin
                
                if cond_touch and cond_hold and cond_vol_shrink and cond_candle:
                     add_signal = True
                     add_reason += "场景 B (趋势回踩: 触碰 MA/守住 MA10 且 缩量); "

                # 场景 C: 反包确认 (日内) - 很难用日线数据完美检查
                # 近似: 昨日收阴, 今日最低触碰 MA, 今日收盘 > 昨日最高
                idx = df.index.get_loc(date)
                if idx > 0:
                    is_prev_green = df.iloc[idx-1]['close'] < df.iloc[idx-1]['open'] # 阴线
                    cond_touch = (row['low'] <= row['MA5']) or (row['low'] <= row['MA10'])
                    
                    if is_prev_green and cond_touch and (row['close'] > row['prev_high']):
                        add_signal = True
                        add_reason += "场景 C (反包确认); "

                if add_signal:
                    position += 1
                    holdings_pct = min(80, holdings_pct + 30)
                    logs.append(f"[{date_str}] 加仓 (+30%): {add_reason} 新仓位: {holdings_pct}% 价格: {row['close']:.2f}")

    # 输出最终状态
    print("\n--- 策略执行日志 ---")
    for log in logs:
        print(log)
        
    print("\n--- 当前状态 ---")
    if analysis_df.empty:
        print("无数据可用。")
        return
        
    latest = df.iloc[-1]
    latest_date = latest.name.strftime("%Y-%m-%d")
    print(f"日期: {latest_date}")
    print(f"价格: {latest['close']:.2f}")
    print(f"MA5: {latest['MA5']:.2f}, MA10: {latest['MA10']:.2f}, MA20: {latest['MA20']:.2f}")
    
    if position == 0:
        print("仓位: 空仓 (等待信号)")
    else:
        print(f"仓位: {holdings_pct}% (等级 {position})")
        if entry_date:
            print(f"建仓日期: {entry_date.strftime('%Y-%m-%d')}")
            pnl = (latest['close'] - entry_price) / entry_price * 100
            print(f"预估盈亏 (底仓): {pnl:.2f}%")
        else:
            print("建仓日期: N/A")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查焦佬策略 (股票)")
    parser.add_argument("code", nargs='?', default=DEFAULT_STOCK_CODE, help=f"股票代码 (6位) [默认: {DEFAULT_STOCK_CODE}]")
    parser.add_argument("-d", "--date", default=DEFAULT_MENTION_DATE, help=f"首次提及日期 (YYYY-MM-DD 或 MM.DD) [默认: {DEFAULT_MENTION_DATE}]")
    parser.add_argument("-p", "--platform", type=int, default=PLATFORM_DAYS_DEFAULT, help=f"平台天数 (默认: {PLATFORM_DAYS_DEFAULT})")
    
    args = parser.parse_args()
    
    # 处理 MM.DD 日期格式为当前年份
    date_input = args.date
    if date_input and len(date_input.split('.')) == 2:
        # 简单转换为当前年份 YYYY-MM-DD
        import datetime
        year = datetime.datetime.now().year
        m, d = date_input.split('.')
        date_input = f"{year}-{m}-{d}"
        
    check_strategy(args.code, date_input, args.platform)
