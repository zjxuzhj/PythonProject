import os
from datetime import datetime

import pandas as pd

import getAllStockCsv
import first_limit_up_ma5_normal as normal
from first_limit_up_ma5_normal import StrategyConfig


def _normalize_digits(code):
    return ''.join(filter(str.isdigit, str(code)))


def _is_bj_stock(code):
    code_str = str(code).lower()
    digits = _normalize_digits(code_str)
    return code_str.startswith('bj') or digits.startswith(('4', '8', '92'))


def _classify_pattern_type(df: pd.DataFrame, first_idx: int, second_idx: int):
    if second_idx - first_idx <= 1:
        return "非W"
    mid = df['close'].iloc[first_idx + 1:second_idx]
    if mid.empty:
        return "非W"
    mid_max = mid.max()
    if pd.isna(mid_max):
        return "非W"
    first_close = df['close'].iloc[first_idx]
    second_close = df['close'].iloc[second_idx]
    return "W底" if mid_max > max(first_close, second_close) else "非W"


def _is_downtrend(close, ma10, ma20, idx: int):
    if idx < 3:
        return False
    c = close.iloc[idx]
    m10 = ma10.iloc[idx]
    m20 = ma20.iloc[idx]
    m20_prev = ma20.iloc[idx - 3]
    if pd.isna(c) or pd.isna(m10) or pd.isna(m20) or pd.isna(m20_prev):
        return False
    return c < m20 and m10 < m20 and m20 < m20_prev


def _find_second_volume_shrink(df: pd.DataFrame, recent_days: int = 5, second_vs_prev_ratio: float = 0.5, require_limit_up_between: bool = False):
    if df is None or df.empty:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if len(df) < 6:
        return None
    required_cols = {'volume', 'close', 'low'}
    if not required_cols.issubset(df.columns):
        return None
    vol = df['volume']
    close = df['close']
    low = df['low']
    high = df['high'] if 'high' in df.columns else None
    prev_close = close.shift(1)
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()
    first_candidates = vol < vol_ma5
    if not first_candidates.any():
        return None
    last_days = df.index[-recent_days:] if len(df) >= recent_days else df.index
    last_set = set(last_days)
    for idx2 in range(len(df)):
        date2 = df.index[idx2]
        if date2 not in last_set:
            continue
        vol2 = vol.iloc[idx2]
        if pd.isna(vol2):
            continue
        if idx2 < 1:
            continue
        prev_vol = vol.iloc[idx2 - 1]
        if pd.isna(prev_vol) or prev_vol <= 0:
            continue
        if vol2 > prev_vol * second_vs_prev_ratio:
            continue
        if vol2 >= vol_ma5.iloc[idx2]:
            continue
        close2 = close.iloc[idx2]
        if pd.isna(close2):
            continue
        if _is_downtrend(close, ma10, ma20, idx2):
            continue
        limit_up2 = False
        limit_down2 = False
        if 'is_limit' in df.columns:
            limit_up2 = bool(df['is_limit'].iloc[idx2])
        if 'is_down_limit' in df.columns:
            limit_down2 = bool(df['is_down_limit'].iloc[idx2])
        if limit_up2 or limit_down2:
            continue
        # Collect valid candidates for first shrink
        valid_first_candidates = []
        for idx1 in range(idx2 - 1, -1, -1):
            if not first_candidates.iloc[idx1]:
                continue
            vol1 = vol.iloc[idx1]
            if pd.isna(vol1):
                continue
            if idx1 < 4:
                continue
            if (idx2 - idx1) < 3:
                continue
            if (idx2 - idx1) > 10:
                continue
            # Relaxed condition: No strict volume comparison vol2 vs vol1
            # Relaxed condition: No strict close1 == min check
            
            if _is_downtrend(close, ma10, ma20, idx1):
                continue
            has_volume_spike = False
            has_price_spike = False
            has_limit_up_between = False
            
            if high is not None and idx2 - idx1 > 1:
                between_start = idx1 + 1
                between_end = idx2
                vol_window = vol.iloc[between_start:between_end]
                ma5_window = vol_ma5.iloc[between_start:between_end]
                ma20_window = vol_ma20.iloc[between_start:between_end]
                high_window = high.iloc[between_start:between_end]
                prev_close_window = prev_close.iloc[between_start:between_end]
                
                if not vol_window.empty:
                    has_volume_spike = ((vol_window > ma5_window) & (vol_window > ma20_window)).any()
                if not high_window.empty:
                    has_price_spike = ((high_window - prev_close_window) / prev_close_window > 0.045).any()
                
                if 'is_limit' in df.columns:
                     limit_window = df['is_limit'].iloc[between_start:between_end]
                     has_limit_up_between = limit_window.any()

            if require_limit_up_between:
                if not has_limit_up_between:
                    continue
            else:
                if not (has_volume_spike and has_price_spike):
                    continue
            limit_up1 = False
            limit_down1 = False
            if 'is_limit' in df.columns:
                limit_up1 = bool(df['is_limit'].iloc[idx1])
            if 'is_down_limit' in df.columns:
                limit_down1 = bool(df['is_down_limit'].iloc[idx1])
            if limit_up1 or limit_down1:
                continue
            low1 = df['low'].iloc[idx1]
            if pd.isna(low1):
                continue
            close1 = df['close'].iloc[idx1]
            if pd.isna(close1):
                continue
            # Removed strict min close check
            if close2 < low1:
                continue
            
            # Check: Max close between first and second shrink >= Max close of 5 days before first shrink * 0.98
            mid_prices = close.iloc[idx1 + 1 : idx2]
            if mid_prices.empty:
                continue
            mid_max_close = mid_prices.max()
            
            pre_start = max(0, idx1 - 5)
            pre_prices = close.iloc[pre_start : idx1]
            if pre_prices.empty:
                # Should not happen given idx1 >= 4 check, but safe to handle
                continue
            pre_max_close = pre_prices.max()
            
            if mid_max_close < pre_max_close * 0.98:
                continue
            
            # Check: High of second shrink day cannot be the highest of the period [idx1, idx2]
            # It must be strictly lower than the max high of the previous days in the interval
            period_highs = high.iloc[idx1 : idx2]
            if not period_highs.empty:
                if high.iloc[idx2] >= period_highs.max():
                    continue

            valid_first_candidates.append({
                'idx1': idx1,
                'vol': vol1,
                'date': df.index[idx1]
            })

        if not valid_first_candidates:
            continue
        
        # Select best candidate: lowest volume
        best_candidate = sorted(valid_first_candidates, key=lambda x: x['vol'])[0]
        idx1 = best_candidate['idx1']
        vol1 = best_candidate['vol']
        
        # Calculate Score
        score = 0
        # Condition 1: Low <= MA20 and Close >= MA20 (+10)
        if low.iloc[idx2] <= ma20.iloc[idx2] and close.iloc[idx2] >= ma20.iloc[idx2]:
            score += 10
        
        # Condition 2: Low <= MA5 and Close >= MA5 (+10)
        if low.iloc[idx2] <= ma5.iloc[idx2] and close.iloc[idx2] >= ma5.iloc[idx2]:
            score += 10
            
        # Condition 3: Vol < MA20_vol (+10)
        if vol2 < vol_ma20.iloc[idx2]:
            score += 10
            
        # Condition 4: Vol[i-2] > Vol[i-1] > Vol[i] (+10)
        if idx2 >= 2:
            v_prev1 = vol.iloc[idx2 - 1]
            v_prev2 = vol.iloc[idx2 - 2]
            if v_prev2 > v_prev1 > vol2:
                score += 10

        pattern = _classify_pattern_type(df, idx1, idx2)
        return {
            'first_idx': idx1,
            'second_idx': idx2,
            'first_date': df.index[idx1],
            'second_date': date2,
            'first_volume': vol1,
            'second_volume': vol2,
            'pattern': pattern,
            'score': score
        }
    return None


def _add_limit_flags(df: pd.DataFrame, symbol: str, config: StrategyConfig):
    if df is None or df.empty:
        return df
    limit_rate = config.MARKET_LIMIT_RATES[normal.get_market_type(symbol)]
    df['prev_close'] = df['close'].shift(1)
    df['limit_price'] = (df['prev_close'] * (1 + limit_rate)).round(2)
    df['is_limit'] = df['close'] >= df['limit_price']
    df['down_limit_price'] = (df['prev_close'] * (1 - limit_rate)).round(2)
    df['is_down_limit'] = df['close'] <= df['down_limit_price']
    return df


def _calc_turnover_rate(volume_value, float_shares_wan):
    try:
        vol = float(volume_value)
        fs = float(float_shares_wan)
        if fs <= 0:
            return None
        if fs > 1_000_000:
            fs = fs / 10000.0
        return round(vol / fs, 2)
    except Exception:
        return None


def _calc_close_increase(df: pd.DataFrame, start_idx: int, day_offset: int):
    """
    计算从start_idx后第day_offset天的收盘价涨幅
    涨幅定义为：(第day_offset天收盘价 - start_idx日收盘价) / start_idx日收盘价 * 100
    """
    target_idx = start_idx + day_offset
    if target_idx >= len(df):
        return None
        
    base_price = df['close'].iloc[start_idx]
    if base_price <= 0:
        return None
        
    target_price = df['close'].iloc[target_idx]
    if pd.isna(target_price):
        return None
        
    increase = (target_price - base_price) / base_price * 100
    return round(increase, 2)


def _calc_sector_avg_increase(sector_codes: list, query_tool, config, target_date, days=3):
    """
    计算板块内所有股票在target_date之前days天的平均涨幅
    """
    total_increase = 0
    valid_count = 0
    
    for code in sector_codes:
        if _is_bj_stock(code):
            continue
        # 这里为了效率，可能需要优化，比如预加载或缓存
        # 但考虑到板块内股票数量有限（通常几十只），且外层循环已经过滤了很多，暂时直接获取
        # 注意：get_stock_data 可能会比较慢，如果有缓存会更好
        df, _ = normal.get_stock_data(code, config)
        if df is None or df.empty:
            continue
            
        # 确保日期索引
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # 找到 target_date 在 df 中的位置
        # 如果 target_date 不在 df 中（比如停牌），找最近的一个交易日
        if target_date not in df.index:
             # 简单处理：如果找不到当天，就跳过
             continue
             
        idx = df.index.get_loc(target_date)
        if idx < days:
            continue
            
        # 计算过去days天的涨幅: (Close[i] - Close[i-days]) / Close[i-days]
        # 或者使用区间涨幅
        # 这里按题目要求：前3个平均涨幅。通常指前3天的累计涨幅
        prev_close = df['close'].iloc[idx - days]
        curr_close = df['close'].iloc[idx]
        
        if prev_close > 0:
            increase = (curr_close - prev_close) / prev_close * 100
            total_increase += increase
            valid_count += 1
            
    if valid_count == 0:
        return 0.0
    return total_increase / valid_count


def scan_second_volume_shrink(recent_days: int = 5, second_vs_prev_ratio: float = 0.9, require_limit_up_between: bool = False):
    config = StrategyConfig()
    query_tool = getAllStockCsv.StockQuery()
    stock_list = query_tool.get_all_filter_stocks()[['stock_code', 'stock_name']].values
    results = []
    
    # 预加载板块成分股缓存，避免重复查询
    # l2_name -> [code1, code2, ...]
    # 注意：这里我们无法预计算板块涨幅，因为每只股票的触发日期(second_date)可能不同
    # 所以必须在循环内部针对特定的日期计算板块涨幅
    
    print(f"扫描参数: 最近{recent_days}天, 缩量比例{second_vs_prev_ratio}, 中间涨停要求: {'是' if require_limit_up_between else '否'}")

    for code, name in stock_list:
        if _is_bj_stock(code):
            continue
        df, _ = normal.get_stock_data(code, config)
        if df is None or df.empty:
            continue
        df = normal.filter_df_to_trading_days_local(df)
        if df is None or df.empty:
            continue
        df = _add_limit_flags(df, code, config)
        float_shares = query_tool.get_stock_float_shares(code)
        match = _find_second_volume_shrink(
            df,
            recent_days=recent_days,
            second_vs_prev_ratio=second_vs_prev_ratio,
            require_limit_up_between=require_limit_up_between
        )
        if match is None:
            continue
        
        # --- 板块过滤逻辑 ---
        # 1. 获取申万二级行业
        _, sw_l2, _ = query_tool.get_sw_industry(code)
        if not sw_l2: # 如果没有行业信息，默认保留或剔除？这里选择保留但标记
            sector_avg_inc = 0.0
        else:
            # 2. 获取该板块成分股
            sector_codes = query_tool.get_sw_l2_constituents(sw_l2)
            # 3. 计算板块在触发日(second_date)前3天的平均涨幅
            # 注意：这里计算的是触发日当天的板块表现（相对于3天前）
            trigger_date = match['second_date']
            sector_avg_inc = _calc_sector_avg_increase(sector_codes, query_tool, config, trigger_date, days=3)
            
            # 4. 过滤条件：板块平均涨幅 > 1.5%
            if sector_avg_inc <= 1.5:
                continue
        # -------------------

        first_idx = match['first_idx']
        second_idx = match['second_idx']
        
        if first_idx >= 4:
            volumes = df['volume'].iloc[first_idx - 4:first_idx + 1].tolist()
            turnovers = [_calc_turnover_rate(v, float_shares) for v in volumes]
            if all(t is not None and t < 1 for t in turnovers):
                continue
            low_turnover_count = sum(1 for t in turnovers if t is not None and t < 1)
            if low_turnover_count >= 3:
                continue
        first_turnover = _calc_turnover_rate(match['first_volume'], float_shares)
        second_turnover = _calc_turnover_rate(match['second_volume'], float_shares)
        
        # 计算后续收盘价涨幅
        inc_1d = _calc_close_increase(df, second_idx, 1)
        inc_2d = _calc_close_increase(df, second_idx, 2)
        inc_3d = _calc_close_increase(df, second_idx, 3)
        
        results.append({
            "代码": code,
            "名称": name,
            "行业": sw_l2,  # 新增行业列
            "板块3日涨幅(%)": round(sector_avg_inc, 2), # 新增板块涨幅列
            "触发日期": match['second_date'].strftime('%Y-%m-%d'),
            "第一次缩量日期": match['first_date'].strftime('%Y-%m-%d'),
            "第一次缩量量能": match['first_volume'],
            "第二次缩量日期": match['second_date'].strftime('%Y-%m-%d'),
            "第二次缩量量能": match['second_volume'],
            "第一次换手率(%)": first_turnover,
            "第二次换手率(%)": second_turnover,
            "形态类型": match['pattern'],
            "分数": match['score'],
            "1日后收盘涨幅(%)": inc_1d if inc_1d is not None else "-",
            "2日后收盘涨幅(%)": inc_2d if inc_2d is not None else "-",
            "3日后收盘涨幅(%)": inc_3d if inc_3d is not None else "-"
        })
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(by=["触发日期", "分数"], ascending=[False, False])
    return result_df


def _calc_stats(result_df: pd.DataFrame):
    if result_df.empty:
        return

    print("\n" + "="*50)
    print("统计结果 (胜率: 涨幅>0的占比, 盈亏比: 平均盈利/平均亏损)")
    print("="*50)

    for day in [1, 2, 3]:
        col_name = f"{day}日后收盘涨幅(%)"
        if col_name not in result_df.columns:
            continue
            
        # 过滤掉无效数据
        valid_data = result_df[result_df[col_name] != "-"]
        if valid_data.empty:
            print(f"{day}日后: 无有效数据")
            continue
            
        values = pd.to_numeric(valid_data[col_name])
        total_count = len(values)
        win_count = len(values[values > 0])
        loss_count = len(values[values < 0])
        
        win_rate = (win_count / total_count * 100) if total_count > 0 else 0
        
        avg_win = values[values > 0].mean() if win_count > 0 else 0
        avg_loss = abs(values[values < 0].mean()) if loss_count > 0 else 0
        
        pl_ratio = (avg_win / avg_loss) if avg_loss > 0 else float('inf')
        pl_ratio_str = f"{pl_ratio:.2f}" if avg_loss > 0 else "Inf"
        
        print(f"{day}日后: 样本数={total_count}, 胜率={win_rate:.2f}%, 盈亏比={pl_ratio_str} (平均盈利={avg_win:.2f}%, 平均亏损={avg_loss:.2f}%)")
    print("="*50)

if __name__ == '__main__':
    # 默认不强制要求涨停，保持原逻辑
    # 如果需要强制要求中间有涨停，请将 require_limit_up_between 设为 True
    result_df = scan_second_volume_shrink(recent_days=5, require_limit_up_between=True)
    if result_df.empty:
        print("最近5个交易日未发现符合条件的二次缩量股票")
    else:
        print(result_df.to_string(index=False))
        print(f"\n总数: {len(result_df)}只股票")
        _calc_stats(result_df)
