import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import datetime, timedelta

# 策略配置类
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
                # print(f"已从缓存加载数据: {filename}")
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
    
    # 计算均线
    for ma in [20, 30, 60, 120, 250]:
        df[f'MA{ma}'] = df['close'].rolling(window=ma).mean()
        
    # 计算成交量均线
    df['MA_Vol_5'] = df['volume'].rolling(window=5).mean()
    df['MA_Vol_20'] = df['volume'].rolling(window=20).mean()
        
    return df

def find_gaps(df, current_date, lookback_days=120):
    """
    寻找跳空缺口（支撑位）
    逻辑：
    1. 向上跳空：Open(T) > Close(T-1)
    2. 有效性：缺口之后的价格没有完全回补（即后续最低价没有低于 Close(T-1)）
    """
    supports = []
    
    # 截取分析区间
    start_date = current_date - timedelta(days=lookback_days)
    mask = (df.index >= start_date) & (df.index <= current_date)
    analysis_df = df.loc[mask].copy()
    
    if len(analysis_df) < 2:
        return supports
    
    # 计算缺口
    analysis_df['prev_close'] = analysis_df['close'].shift(1)
    analysis_df['gap_size'] = analysis_df['open'] - analysis_df['prev_close']
    
    # 计算涨跌幅 (相对于前一天的收盘价)
    analysis_df['pct_change'] = analysis_df['close'].pct_change()
    
    # 筛选向上跳空缺口
    gap_days = analysis_df[analysis_df['gap_size'] > 0].index
    
    # 获取所有的日期索引
    all_dates = analysis_df.index
    
    for gap_date in gap_days:
        gap_row = analysis_df.loc[gap_date]
        gap_low = gap_row['prev_close']  # 缺口下沿（最强支撑）
        gap_high = gap_row['open']       # 缺口上沿
        
        # 检查当天最低价是否已经回补了缺口
        # 如果 Low <= prev_close，说明当天就回补了，不算有效跳空支撑
        if gap_row['low'] <= gap_low:
            continue
            
        # 获取前一天的日期
        idx = all_dates.get_loc(gap_date)
        if idx == 0: # 如果是第一天，无法获取前一天，跳过
            continue
        prev_date = all_dates[idx-1]
        
        # 检查前一天的涨幅是否超过 5%
        # 注意：我们需要前一天的涨跌幅数据
        prev_pct_change = analysis_df.loc[prev_date, 'pct_change']
        
        # 如果前一天涨幅 <= 5%，则跳过该缺口
        if pd.isna(prev_pct_change) or prev_pct_change <= 0.05:
            continue
            
        # 检查缺口是否被回补
        # 获取缺口日期之后的所有数据
        future_data = analysis_df[analysis_df.index > gap_date]
        
        if future_data.empty:
            is_valid = True
        else:
            # 修改逻辑：使用收盘价判断缺口是否被回补
            # 如果后续收盘价低于缺口下沿，则视为已完全回补，支撑失效
            # 用户偏好：即使盘中跌破（Low < Gap Low），只要收盘收回（Close > Gap Low），仍视为有效支撑
            # 进一步优化：允许 1% 的容差，防止因为轻微跌破（假摔）而漏掉重要支撑
            min_close = future_data['close'].min()
            is_valid = min_close > gap_low * 0.99
            
        if is_valid:
            # 权重计算
            weight = 5
            desc = f"{prev_date.strftime('%Y-%m-%d')} (涨幅{prev_pct_change:.2%}) 至 {gap_date.strftime('%Y-%m-%d')} 形成向上跳空缺口，支撑区间 [{gap_low:.2f}, {gap_high:.2f}]"
            
            # 检查是否有均线支撑确认
            # 遍历缺口形成后的每一天，检查是否有均线回踩确认
            # 确认条件：
            # 1. 均线值与缺口下沿非常接近（例如 +/- 2%）
            # 2. 当天最低价触及均线（Low <= MA * 1.01）
            # 3. 当天收盘价站稳均线（Close >= MA）
            ma_confirmations = []
            if not future_data.empty:
                for ma_window in [20, 30, 60, 120, 250]:
                    ma_col = f'MA{ma_window}'
                    if ma_col not in future_data.columns:
                        continue
                        
                    found_for_this_ma = False
                    for date, row in future_data.iterrows():
                        ma_val = row[ma_col]
                        if pd.isna(ma_val):
                            continue
                            
                        # 检查均线是否在缺口附近
                        if abs(ma_val - gap_low) / gap_low <= 0.02:
                            # 检查是否获得支撑
                            if row['low'] <= ma_val * 1.01 and row['close'] >= ma_val:
                                ma_confirmations.append(f"{date.strftime('%Y-%m-%d')}获{ma_window}日线({ma_val:.2f})支撑")
                                weight += 1 # 每确认一次增加权重
                                found_for_this_ma = True
                                break # 每个均线只记录一次最近的确认
                    if found_for_this_ma:
                        continue
                
                if ma_confirmations:
                    desc += f" | 共振: {'; '.join(ma_confirmations)}"
            
            supports.append({
                'date': gap_date,
                'type': '跳空缺口',
                'price': gap_low,
                'range': (gap_low, gap_high),
                'weight': weight, # 更新后的权重
                'desc': desc
            })
            
    # 按价格从高到低排序（离当前价格最近的支撑在前）
    supports.sort(key=lambda x: x['price'], reverse=True)
    return supports

def find_resistances(df, current_date, lookback_days=60):
    """
    寻找压力位
    逻辑：
    1. 近期高点：区间内的最高价
    2. 上影线密集区：
       - 上影线较长
       - 密集出现
       - 取密集区内的最高收盘价
    """
    resistances = []
    
    # 截取分析区间
    start_date = current_date - timedelta(days=lookback_days)
    mask = (df.index >= start_date) & (df.index <= current_date)
    analysis_df = df.loc[mask].copy()
    
    if len(analysis_df) < 5:
        return resistances
    
    # 1. 寻找波段高点 (Swing Highs)
    # 简单逻辑：过去N天的最高点
    global_high = analysis_df['high'].max()
    global_high_date = analysis_df['high'].idxmax()
    
    resistances.append({
        'date': global_high_date,
        'type': '近期高点',
        'price': global_high,
        'weight': 3,
        'desc': f"{global_high_date.strftime('%Y-%m-%d')} 创下阶段高点 {global_high:.2f}"
    })
    
    # 2. 上影线密集区
    # 计算上影线长度
    analysis_df['upper_shadow'] = analysis_df['high'] - analysis_df[['open', 'close']].max(axis=1)
    analysis_df['body'] = (analysis_df['close'] - analysis_df['open']).abs()
    analysis_df['price'] = analysis_df['close']
    
    # 定义长上影线：上影线长度 > 实体长度 * 0.5 (或者 > 股价 * 1%)
    # 这里使用相对宽松的定义以捕捉更多特征
    analysis_df['is_long_upper'] = (analysis_df['upper_shadow'] > analysis_df['body'] * 0.5) & \
                                   (analysis_df['upper_shadow'] > analysis_df['price'] * 0.005)
    
    # 寻找密集区 (Rolling Window)
    window_size = 5
    threshold = 3 # 5天内有3天长上影线
    
    # 遍历寻找
    # 为了避免重复，我们找到一个区间后跳过
    i = 0
    dates = analysis_df.index
    while i < len(dates) - window_size + 1:
        window_dates = dates[i : i + window_size]
        window_data = analysis_df.loc[window_dates]
        
        count = window_data['is_long_upper'].sum()
        
        if count >= threshold:
            # 找到密集区
            # 压力位1：最高收盘价 (用户提到的 "最高收盘价")
            max_close = window_data['close'].max()
            max_close_date = window_data['close'].idxmax()
            
            # 压力位2：区间最高点 (通常已被 "近期高点" 覆盖，但如果是局部高点也有意义)
            local_high = window_data['high'].max()
            
            # 避免与全局高点重复太近 (价格差异 < 1%)
            is_duplicate_high = abs(local_high - global_high) / global_high < 0.01
            
            resistances.append({
                'date': window_dates[-1], # 标记为区间结束日
                'type': '上影线密集区(收盘)',
                'price': max_close,
                'weight': 4,
                'desc': f"{window_dates[0].strftime('%Y-%m-%d')} 至 {window_dates[-1].strftime('%Y-%m-%d')} 上影线密集，最高收盘价 {max_close:.2f}"
            })
            
            if not is_duplicate_high:
                 resistances.append({
                    'date': window_dates[-1],
                    'type': '上影线密集区(最高)',
                    'price': local_high,
                    'weight': 3,
                    'desc': f"{window_dates[0].strftime('%Y-%m-%d')} 至 {window_dates[-1].strftime('%Y-%m-%d')} 上影线密集，最高价 {local_high:.2f}"
                })
            
            i += window_size # 跳过该区间
        else:
            i += 1
            
    # 按价格排序
    resistances.sort(key=lambda x: x['price'])
    return resistances

def find_dense_touch_levels(df, current_date, lookback_days=60):
    """
    寻找密集接触点（支撑/压力位）
    逻辑：分别统计 Low 的密集区（支撑）和 High 的密集区（压力）
    """
    # 截取分析区间
    start_date = current_date - timedelta(days=lookback_days)
    mask = (df.index >= start_date) & (df.index <= current_date)
    analysis_df = df.loc[mask].copy()
    
    if len(analysis_df) < 5:
        return []
        
    levels = []
    
    # 1. 寻找支撑位 (Low 的密集区)
    low_counts = {}
    low_dates = {} # 记录每个价格点对应的日期列表
    
    for date, row in analysis_df.iterrows():
        val = int(round(row['low'] * 100))
        # 增加模糊性，例如 +/- 3 分钱都算命中，以捕捉用户视角的"底部区域"
        for p in range(val - 3, val + 4):
            low_counts[p] = low_counts.get(p, 0) + 1
            if p not in low_dates:
                low_dates[p] = []
            low_dates[p].append(date.strftime('%Y-%m-%d'))
            
    # 排序并筛选 Top 5
    if low_counts:
        sorted_lows = sorted(low_counts.items(), key=lambda x: x[1], reverse=True)
        processed_prices = set()
        
        for price_int, count in sorted_lows:
            # 降低门槛：至少要有 2 天触及 (或总天数的 5%)
            if count < max(2, len(analysis_df) * 0.05):
                break
                
            if price_int in processed_prices:
                continue
                
            # 标记相邻价格 (扩大排重范围)
            for p in range(price_int - 5, price_int + 6):
                processed_prices.add(p)
                
            price_val = price_int / 100.0
            
            # 获取该价格点对应的日期，并去重
            dates_list = sorted(list(set(low_dates[price_int])))
            
            levels.append({
                'price': price_val,
                'count': count, # 这里是"模糊命中"的次数
                'type': 'support', # 标记为支撑倾向
                'total_days': len(analysis_df),
                'ratio': count / len(analysis_df),
                'dates': dates_list # 保存完整日期列表
            })
            if len(levels) >= 5: # 取前5个支撑
                break

    # 2. 寻找压力位 (High 的密集区)
    high_counts = {}
    high_dates = {}
    
    for date, row in analysis_df.iterrows():
        val = int(round(row['high'] * 100))
        for p in range(val - 3, val + 4): # 同样扩大模糊范围
            high_counts[p] = high_counts.get(p, 0) + 1
            if p not in high_dates:
                high_dates[p] = []
            high_dates[p].append(date.strftime('%Y-%m-%d'))
            
    # 排序并筛选 Top 5
    if high_counts:
        sorted_highs = sorted(high_counts.items(), key=lambda x: x[1], reverse=True)
        processed_prices = set() 
        
        count_added = 0
        for price_int, count in sorted_highs:
            if count < max(2, len(analysis_df) * 0.05):
                break
            
            if price_int in processed_prices:
                continue
                
            for p in range(price_int - 5, price_int + 6):
                processed_prices.add(p)
                
            price_val = price_int / 100.0
            
            dates_list = sorted(list(set(high_dates[price_int])))
            
            levels.append({
                'price': price_val,
                'count': count,
                'type': 'resistance', # 标记为压力倾向
                'total_days': len(analysis_df),
                'ratio': count / len(analysis_df),
                'dates': dates_list
            })
            count_added += 1
            if count_added >= 5:
                break
                
    return levels

def find_breakout_supports(df, current_date, lookback_days=60, breakout_window=30):
    """
    寻找"压力转支撑"的位点 (顶底转换)
    逻辑：
    1. 寻找过去一段时间 (lookback_days) 内的波段高点/平台压力位
    2. 检查最近 (breakout_window) 是否有放量突破行为
    3. 确认突破后未被有效跌破
    """
    supports = []
    
    # 截取数据
    start_date = current_date - timedelta(days=lookback_days)
    mask = (df.index >= start_date) & (df.index <= current_date)
    analysis_df = df.loc[mask].copy()
    
    if len(analysis_df) < 10:
        return supports
        
    # 1. 寻找潜在压力位 (过去的高点)
    # 我们关注 breakout_window 之前的高点
    # 用户案例：12-25 是高点，01-06 突破。当前日期 01-19。
    # 01-06 距离 01-19 是 13 天。
    # 12-25 距离 01-19 是 25 天。
    # 所以 breakout_window 设为 20，breakout_start_date 是 12-30。
    # past_df 是 12-30 之前的数据，包含了 12-25。这是对的。
    # 但如果 breakout_window 太短，可能漏掉突破点。
    # 如果 breakout_window 太长，可能 past_df 太短。
    
    # 动态调整：我们不仅关注 breakout_start_date 之前的高点，
    # 其实只要是 current_date 之前的高点，且被最近的 K 线突破了，都算。
    # 但是为了定义"顶底转换"，通常意味着这个高点形成了一段时间的阻力。
    
    # 修改逻辑：
    # 1. 遍历最近 20 天 (breakout_window) 的每一天作为潜在的"突破日"
    # 2. 对于每个潜在突破日，检查它是否突破了"之前" (比如过去 60 天内) 的某个高点
    
    recent_breakout_mask = (analysis_df.index >= current_date - timedelta(days=breakout_window))
    recent_breakout_df = analysis_df.loc[recent_breakout_mask]
    
    # past_high_df 应该是相对于 breakout 日期之前的，但为了简化计算，我们取 breakout_window 之前的数据作为"确定的过去"
    # 也可以取整个 analysis_df，但在 loop 中判断 p_date < r_date
    
    # 我们需要确保 analysis_df 包含足够长的时间来寻找高点
    # 如果 lookback_days=60，那么数据从 11-19 开始。
    # 12-25 在范围内。
    
    all_potential_resistances = []
    
    # 寻找局部高点 (Window=3)
    window = 3
    # 确保有足够数据
    if len(analysis_df) > window * 2:
        for i in range(window, len(analysis_df) - window):
            center_idx = analysis_df.index[i]
            center_high = analysis_df.iloc[i]['high']
            center_close = analysis_df.iloc[i]['close']
            center_open = analysis_df.iloc[i]['open']
            
            # 必须是局部高点
            is_local_max = True
            for k in range(1, window + 1):
                if analysis_df.iloc[i-k]['high'] > center_high or analysis_df.iloc[i+k]['high'] > center_high:
                    is_local_max = False
                    break
            
            if is_local_max:
                 all_potential_resistances.append((center_high, center_idx, "波段高点"))
                 
                 # 同时也加入收盘价高点 (如果差异大)
                 if abs(center_close - center_high) / center_high > 0.01:
                      all_potential_resistances.append((center_close, center_idx, "收盘高点"))
                 
                 # 特殊处理：加入 Open/Close 的最大值作为潜在压力 (针对长上影线情况)
                 # 例如 12-25 Open=7.82, High=7.82 (T字板或光头)，这里 High=Open
                 body_top = max(center_open, center_close)
                 if abs(body_top - center_high) / center_high > 0.005 and abs(body_top - center_close) > 0.01:
                      all_potential_resistances.append((body_top, center_idx, "实体顶部压力"))

    # 现在检查突破
    # 遍历 recent_breakout_df 中的每一天
    processed_prices = set()
    
    current_price = df.loc[current_date]['close'] # 重新获取当前价格
    
    for r_date, r_row in recent_breakout_df.iterrows():
        # 检查这一天是否放量/大阳线
        is_volume_up = False
        vol_ratio = 0
        if 'MA_Vol_20' in df.columns:
              ma_vol = df.loc[r_date, 'MA_Vol_20']
              if pd.notna(ma_vol) and ma_vol > 0:
                  vol_ratio = r_row['volume'] / ma_vol
                  if vol_ratio > 1.3:
                      is_volume_up = True
        
        is_price_up = r_row['close'] / r_row['open'] > 1.03
        is_gap_up = False
        prev_idx = df.index.get_loc(r_date) - 1
        if prev_idx >= 0:
             prev_date = df.index[prev_idx]
             if df.loc[prev_date, 'high'] < r_row['open']:
                  is_gap_up = True
        
        if not (is_volume_up or is_price_up or is_gap_up):
            continue
            
        # 如果这一天是强势日，检查它是否突破了之前的某个高点
        for price, p_date, type_name in all_potential_resistances:
            # 必须是之前的点
            if p_date >= r_date:
                continue
                
            # 必须是最近才突破的 (即 r_date 当天突破，或者 r_date 之前没突破)
            # 简化逻辑：r_row['close'] > price，且 r_row['open'] < price (或者 gap up)
            # 且 price 在 processed_prices 中不存在 (避免重复添加)
            
            if price in processed_prices:
                continue
                
            if r_row['close'] > price and (r_row['open'] < price or is_gap_up) and r_row['close'] > price * 1.005: # 稍微站稳
                 # 找到了突破！
                 
                 # 验证有效性：从突破日之后到当前，没有有效跌破
                 post_breakout_df = analysis_df[analysis_df.index > r_date]
                 is_failed = False
                 if not post_breakout_df.empty:
                     min_close = post_breakout_df['close'].min()
                     if min_close < price * 0.97:
                         is_failed = True
                 
                 if not is_failed and current_price >= price * 0.97:
                     processed_prices.add(price)
                     
                     reasons = []
                     if is_volume_up: reasons.append(f"放量({vol_ratio:.1f}倍)")
                     if is_price_up: reasons.append("大阳线")
                     if is_gap_up: reasons.append("跳空")
                     
                     breakout_desc = f"{r_date.strftime('%Y-%m-%d')} {'+'.join(reasons)}突破"
                     
                     # Calculate time decay weight
                     # 距离突破时间越近，权重越大 (Time Decay)
                     time_diff = (r_date - p_date).days
                     weight = 6
                     if time_diff <= 15:
                         weight += 2
                     elif time_diff <= 30:
                         weight += 1
                     
                     supports.append({
                        'date': r_date, # 突破日作为日期，或者原始高点日期 p_date
                        'type': '顶底转换',
                        'price': price,
                        'range': (price, price),
                        'weight': weight,
                        'p_date': p_date, # 临时存储用于去重
                        'desc': f"{p_date.strftime('%Y-%m-%d')} {type_name} ({price:.2f}) 被 {breakout_desc}，确认顶底转换 (距突破{time_diff}天)"
                    })
    
    # Post-process: 对同一天(p_date)形成的多个支撑位，只保留价格最高的一个作为主支撑，其他的降权
    # 例如：同一天的高点是7.83，收盘是7.70。如果都突破了，7.83作为第一支撑，7.70权重应该降低
    from collections import defaultdict
    grouped_supports = defaultdict(list)
    for s in supports:
        grouped_supports[s['p_date']].append(s)
        
    final_supports = []
    for p_date, group in grouped_supports.items():
        # 按价格从高到低排序
        group.sort(key=lambda x: x['price'], reverse=True)
        
        for i, s in enumerate(group):
            if i > 0:
                s['weight'] = max(1, s['weight'] - 3) # 次级支撑大幅降权
                s['desc'] += " | 同源次级支撑权重降低"
            
            # 移除临时key
            if 'p_date' in s:
                del s['p_date']
            final_supports.append(s)

    return final_supports

def analyze_support_resistance(symbol, date_str):
    print(f"正在分析股票 {symbol} 在 {date_str} 的支撑位和压力位...")
    
    df = get_stock_data(symbol)
    if df is None:
        return

    try:
        target_date = pd.to_datetime(date_str)
    except Exception as e:
        print(f"日期格式错误: {e}")
        return

    # 确保目标日期在数据范围内，或者取最近的数据
    if target_date not in df.index:
        # 寻找目标日期之前最近的一个交易日
        recent_df = df[df.index <= target_date]
        if recent_df.empty:
            print(f"在该日期 {date_str} 之前没有找到数据。")
            return
        real_date = recent_df.index[-1]
        print(f"注意：{date_str} 非交易日或无数据，使用最近交易日 {real_date.strftime('%Y-%m-%d')} 进行分析。")
        target_date = real_date
    
    current_price = df.loc[target_date]['close']
    print(f"当前日期: {target_date.strftime('%Y-%m-%d')}, 收盘价: {current_price:.2f}")
    
    # 计算支撑位和压力位
    supports = find_gaps(df, target_date)
    resistances = find_resistances(df, target_date)
    
    # 寻找顶底转换支撑 (Breakout Support)
    # 使用过去 60 天的高点，检查最近 20 天的突破
    breakout_supports = find_breakout_supports(df, target_date, lookback_days=60, breakout_window=20)
    supports.extend(breakout_supports)
    
    # 使用较短的周期来捕捉近期的密集区 (例如 40-60 天)
    # 用户案例提到的区间约为 40 天 (10.20 - 11.28)
    dense_levels = find_dense_touch_levels(df, target_date, lookback_days=45) 
    
    # 将密集点分类为支撑或压力
    for level in dense_levels:
        price = level['price']
        ratio = level['ratio']
        count = level['count']
        level_type = level.get('type', 'unknown')
        dates = level.get('dates', [])
        
        weight = 3
        if ratio > 0.15: 
            weight += 1
        if ratio > 0.25:
            weight += 1
            
        desc = f"近45日内有 {count} 天 ({ratio:.1%}) 的K线{level_type}触及该价格，构成密集筹码区"
        
        # 检查是否多次跌破 (Penalize frequent breakdowns)
        # 获取近期数据 (与密集区分析周期一致，或稍长)
        check_start_date = target_date - timedelta(days=60)
        recent_mask = (df.index >= check_start_date) & (df.index <= target_date)
        recent_df = df.loc[recent_mask]
        
        if not recent_df.empty:
            # 计算"有效跌破"次数
            # 定义：收盘价跌破支撑位 (Close < Price * 0.995)
            # 使用 crossing logic 避免一次长期下跌被算作多次
            is_close_below = recent_df['close'] < price * 0.995
            is_low_below = recent_df['low'] < price * 0.995
            
            # Shift 1位，填充 False (假设开始时未跌破，或者不计入第一次)
            # 使用 shift(fill_value=False) 直接避免产生 NaN，从而无需 fillna，避免 Downcasting 警告 (Pandas 2.1+)
            was_close_below = is_close_below.shift(1, fill_value=False)
            was_low_below = is_low_below.shift(1, fill_value=False)
            
            # 当天跌破 (is_below) 且 昨天未跌破 (not was_below) -> 视为一次跌破事件
            close_break_events = (is_close_below & ~was_close_below).sum()
            low_break_events = (is_low_below & ~was_low_below).sum()
            
            # 权重扣分逻辑：收盘价跌破权重高(2分)，最低价跌破权重低(1分)
            # 注意：通常收盘价跌破也会触发最低价跌破，所以这里叠加扣分是合理的，或者分开统计
            # 如果收盘价跌破，low_break_events 通常也会 +1，所以 close_break 实际上会扣 3 分 (2+1)
            # 为了符合用户意图 "收盘价跌破权重高点"，我们可以独立计算
            
            penalty_score = 0
            penalty_score += close_break_events * 2  # 收盘价跌破，每次扣2分
            
            # 对于最低价跌破，我们只计算那些 "收盘价没有跌破" 的情况吗？
            # 或者简单点，全部累加。
            # 如果 Close 跌破，Low 肯定跌破。
            # 假设一次 Close 跌破事件：
            #   Low break: +1
            #   Close break: +2
            #   Total: 3
            # 假设一次 Shadow 跌破 (Close > Price, Low < Price):
            #   Low break: +1
            #   Close break: 0
            #   Total: 1
            # 这样符合 "收盘价跌破权重高" 的要求。
            
            penalty_score += low_break_events * 1
            
            if penalty_score >= 6:
                weight -= 4 # 极其严重的破位
                desc += f" | ⚠️严重破位(得分{penalty_score})"
            elif penalty_score >= 3:
                weight -= 2 # 中度破位
                desc += f" | ⚠️多次破位(得分{penalty_score})"
            elif penalty_score >= 1:
                weight -= 1 # 轻微破位
                desc += f" | 有破位记录"
        
        # 检查均线共振和回踩
        # 我们需要遍历 level['dates'] 对应的实际数据
        
        ma_support_desc = []
        for date_str in dates:
            if date_str not in df.index:
                continue
            
            row = df.loc[date_str]
            
            # 检查各条均线
            for ma_window in [20, 30, 60, 120, 250]:
                ma_col = f'MA{ma_window}'
                if ma_col not in df.columns:
                    continue
                
                ma_val = row[ma_col]
                if pd.isna(ma_val):
                    continue
                    
                # 检查均线是否在支撑位附近 (例如 +/- 2%)
                if abs(ma_val - price) / price <= 0.02:
                    # 检查回踩：Low 触及或击穿 MA，Close 站稳
                    if row['low'] <= ma_val * 1.01 and row['close'] >= ma_val:
                        # 检查缩量
                        vol_desc = ""
                        if 'MA_Vol_5' in df.columns and row['volume'] < row['MA_Vol_5']:
                            vol_desc = "缩量"
                            # weight += 1 # 缩量额外加分 (暂时移除)
                            
                        # weight += 1 # 均线支撑加分 (暂时移除，避免权重过高导致误判)
                        ma_support_desc.append(f"{date_str[5:]}{vol_desc}回踩{ma_window}日线({ma_val:.2f})")
        
        if ma_support_desc:
            # 去重并取最后几个
            unique_desc = sorted(list(set(ma_support_desc)))
            # 只显示最近的3个，避免太长
            desc += f" | 共振: {'; '.join(unique_desc[-3:])}"
            # 均线共振权重降低：仅当有 K 线支撑确认后，作为辅助说明，不大幅增加权重
            # 或者仅增加 1 分，保持 K 线形态的主导地位
            weight += 1
            
        if dates:
            # 取最近的3个日期，或者如果少于6个全部显示
            if len(dates) <= 6:
                desc += f" (日期: {', '.join([d[5:] for d in dates])})"
            else:
                desc += f" (最近日期: {', '.join([d[5:] for d in dates[-3:]])}等)"
        
        # 根据类型和当前价格判断
        if level_type == 'support' and price < current_price:
             # 确认为支撑位
            supports.append({
                'date': target_date, 
                'type': '低点密集区',
                'price': price,
                'range': (price, price),
                'weight': weight,
                'desc': desc
            })
        elif level_type == 'resistance' and price > current_price:
             # 确认为压力位
            resistances.append({
                'date': target_date,
                'type': '高点密集区',
                'price': price,
                'weight': weight,
                'desc': desc
            })
        elif price < current_price:
             # 虽然标记为压力，但在当前价格下方，视为支撑
             supports.append({
                'date': target_date, 
                'type': '密集接触区',
                'price': price,
                'range': (price, price),
                'weight': weight - 1, # 权重略降
                'desc': desc
            })
        else:
             # 虽然标记为支撑，但在当前价格上方，视为压力
             resistances.append({
                'date': target_date,
                'type': '密集接触区',
                'price': price,
                'weight': weight - 1,
                'desc': desc
            })

    print("\n" + "="*50)
    print("【支撑位分析 (Support)】")
    print("="*50)
    
    # 过滤掉远低于当前价格的支撑位 (比如低于 20%)，避免输出太多，除非数量很少
    valid_supports = [s for s in supports if s['price'] < current_price * 1.1] # 允许稍微高于现价（如果现价跌破支撑但未完全回补）
    
    # 过滤逻辑
    # 1. 按价格从低到高排序
    valid_supports.sort(key=lambda x: x['price'])
    
    # 2. 找到"底部极限支撑" (价格最低且有一定力度的支撑)
    # 我们认为价格最低的前 2-3 个支撑中，权重最高的那个是"底部基准"
    base_support = None
    if valid_supports:
        # 取价格最低的前3个候选
        candidates = valid_supports[:3]
        # 在候选中找权重最高的，如果权重一样找价格最低的
        base_support = max(candidates, key=lambda x: x['weight'])
        
    filtered_supports = []
    if valid_supports and base_support:
        for s in valid_supports:
            # 规则1: 如果该支撑位价格低于或等于底部基准，保留 (底部支撑)
            if s['price'] <= base_support['price']:
                filtered_supports.append(s)
                continue
                
            # 规则2: 如果该支撑位价格高于底部基准
            # 必须满足: 权重显著高于底部基准 (例如大 2 级以上，或者有均线共振等强逻辑)
            # 且不能是被"明显跌破"的 (虽然 valid_supports 已经过滤了 current_price * 1.1，但这里指相对于当前价格已经失效的中间位)
            
            # 如果当前价格已经跌破了该支撑位 (Price > Current)，通常意味着它变成了压力位，或者已经失效
            # 但 valid_supports 里允许 price < current * 1.1，所以可能包含高于 current 的点
            # 这里我们只关心支撑，即 Price < Current
            
            # 强制过滤: 如果 s['price'] > current_price，它其实是压力位或失效支撑，必须排除
            if s['price'] > current_price:
                continue
                
            # 规则3: 中间位支撑过滤
            # 必须确保没有被"有效跌破"
            # 如果当前价格距离该支撑位很近 (Current < Price * 1.02)，且之前已经跌破过 (High < Price)，则说明已成压力
            # 简单逻辑: 只有当 Price <= Current 时，才保留作为支撑
            # (已通过上面的 if s['price'] > current_price 覆盖)
            
            # 如果是中间位 (Base < Price < Current)
            # 要求权重必须很高，或者距离当前价格比较近 (第一道防线)
            # 否则视作"已被跌破"或"非关键"支撑
            if s['weight'] > base_support['weight'] + 2:
                filtered_supports.append(s)
            
    valid_supports = filtered_supports if filtered_supports else valid_supports

    if not valid_supports:
        print("近期未发现明显缺口支撑。")
    else:
        # 按权重排序 (权重高的在前)，如果权重相同按价格
        valid_supports.sort(key=lambda x: (x['weight'], x['price']), reverse=True)
        
        # 取前 8 个
        for s in valid_supports[:8]:
            range_str = f"[{s['range'][0]:.2f}, {s['range'][1]:.2f}]"
            print(f"价格: {s['price']:.2f} | 权重: {s['weight']} | 类型: {s['type']} | 区间: {range_str}")
            print(f"  说明: {s['desc']}")
            print("-" * 30)

    print("\n" + "="*50)
    print("【压力位分析 (Resistance)】")
    print("="*50)
    
    # 过滤掉远低于当前价格的压力位
    valid_resistances = [r for r in resistances if r['price'] > current_price * 0.9]
    
    if not valid_resistances:
        print("近期未发现明显技术压力位。")
    else:
        # 按价格升序排列（离当前价格最近的压力在前）
        valid_resistances.sort(key=lambda x: x['price'])
        for r in valid_resistances[:5]:
            print(f"价格: {r['price']:.2f} | 权重: {r['weight']} | 类型: {r['type']}")
            print(f"  说明: {r['desc']}")
            print("-" * 30)
            
    print("\n分析完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='股票支撑压力位分析工具')
    parser.add_argument('symbol', nargs='?', default='600879', help='股票代码 (默认: 601179)')
    parser.add_argument('date', nargs='?', default='2025-12-02', help='分析日期 YYYY-MM-DD (默认: 2026-02-06)')
    
    args = parser.parse_args()
    
    analyze_support_resistance(args.symbol, args.date)
