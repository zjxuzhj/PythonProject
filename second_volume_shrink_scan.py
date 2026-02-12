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


def _find_second_volume_shrink(df: pd.DataFrame, recent_days: int = 5, second_vs_prev_ratio: float = 0.5):
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
            if (idx2 - idx1) > 5:
                continue
            # Relaxed condition: No strict volume comparison vol2 vs vol1
            # Relaxed condition: No strict close1 == min check
            
            if _is_downtrend(close, ma10, ma20, idx1):
                continue
            has_volume_spike = False
            has_price_spike = False
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


def scan_second_volume_shrink(recent_days: int = 5, second_vs_prev_ratio: float = 0.9):
    config = StrategyConfig()
    query_tool = getAllStockCsv.StockQuery()
    stock_list = query_tool.get_all_filter_stocks()[['stock_code', 'stock_name']].values
    results = []
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
            second_vs_prev_ratio=second_vs_prev_ratio
        )
        if match is None:
            continue
        first_idx = match['first_idx']
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
        results.append({
            "代码": code,
            "名称": name,
            "触发日期": match['second_date'].strftime('%Y-%m-%d'),
            "第一次缩量日期": match['first_date'].strftime('%Y-%m-%d'),
            "第一次缩量量能": match['first_volume'],
            "第二次缩量日期": match['second_date'].strftime('%Y-%m-%d'),
            "第二次缩量量能": match['second_volume'],
            "第一次换手率(%)": first_turnover,
            "第二次换手率(%)": second_turnover,
            "形态类型": match['pattern'],
            "分数": match['score']
        })
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(by=["触发日期", "分数"], ascending=[False, False])
    return result_df


if __name__ == '__main__':
    result_df = scan_second_volume_shrink(recent_days=5)
    if result_df.empty:
        print("最近5个交易日未发现二次缩量股票")
    else:
        print(result_df.to_string(index=False))
        print(f"\n总数: {len(result_df)}只股票")
