import os

import akshare as ak
import numpy as np
import pandas as pd

import getAllStockCsv as stockCsv


def calculate_returns(df, days_list):
    """预计算各持有期收益率"""
    for days in days_list:
        df[f'return_{days}d'] = df['close'].shift(-days) / df['close'] - 1
    return df


def get_stock_data(symbol, start_date, force_update=False):
    """带本地缓存的数据获取"""
    # 生成唯一文件名（网页1）
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
    try:
        if len(symbol) > 6:
            symbol = stockCsv.StockQuery().get_simple_by_code(symbol)

        """获取日线数据（复权处理）"""
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            adjust="qfq"
        )
        # 数据标准化处理
        df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume'
        }, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        # 保存到本地（网页3）
        os.makedirs("data_cache", exist_ok=True)
        df.set_index('date').to_parquet(  # 保存时带索引[6](@ref)
            cache_path,
            engine='fastparquet',
            compression='snappy'
        )
        print(f"新数据已缓存：{symbol}")

    except Exception as e:
        print(f"数据获取失败：{symbol} - {str(e)}")
        return pd.DataFrame(), False

    return df.set_index('date').sort_index(), False  # 返回API标记


def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()


def calculate_volatility(df, window=20):
    """计算波动性指标"""
    # 振幅指标（网页6）
    df['amplitude'] = (df['high'] - df['low']) / df['open']  # 日振幅率
    df['amplitude_ma'] = df['amplitude'].rolling(window).mean()  # 20日平均振幅

    # 布林带带宽（网页3）
    df['boll_width'] = (df['upper_band'] - df['lower_band']) / df['MA20']
    return df

# 织布机走势判断条件（网页3）
def is_chop_market(df, lookback=30):
    """判断是否织布机行情"""
    chop_cond = (
        (df['amplitude_ma'] < 0.015) &  # 平均振幅<1.5%
        (df['boll_width'] < 0.02)   # 布林带带宽<2%
        # &      # 布林带带宽<2%
        # (df['volume'] < df['volume'].rolling(30).mean().shift(1))  # 成交量低于30日均量
    )
    return chop_cond

def big_red_line_filter(df, window=60):
    """大阴线频率筛选"""
    # 大阴线定义（网页5）
    big_red_cond = (
        (df['close'] < df['open'] * 0.95) # 跌幅>5%
        &
        ((df['open'] - df['close']) > 2 * (df['high'] - df['open']).abs())  # 实体>2倍上影线
    )
    # 统计60日内大阴线次数
    df['big_red_count'] = big_red_cond.rolling(window).sum()
    return df

def calculate_moving_averages(df):
    """计算各类均线指标"""
    # 短期均线 (30天=6周，60天=12周)
    df['MA30'] = df['close'].rolling(30).mean().round(2)
    df['MA60'] = df['close'].rolling(60).mean().round(2)

    # 30周均线（约150个交易日，按5天/周计算）
    df['MA30W'] = df['close'].rolling(30 * 5).mean().round(2)

    # 当前价格与30周均线关系，30周均线反映市场中长期趋势方向，当股价位于其上方时，说明中期趋势未破坏。此时若出现底背离，往往意味着短期调整可能结束，长期趋势将延续
    df['above_30week'] = df['close'] > df['MA30W']  # 价格在均线上方

    # 计算 20 日均线（MA20）
    df['MA20'] = df['close'].rolling(window=20).mean()
    # 计算 20 日标准差
    df['std20'] = df['close'].rolling(window=20).std()
    # 计算布林线上轨和下轨（通常用 2 倍标准差）
    df['upper_band'] = df['MA20'] + 2 * df['std20']  # 确保此步骤已执行
    df['lower_band'] = df['MA20'] - 2 * df['std20']  # 确保此步骤已执行
    return df


def calculate_macd(df, fast=5, slow=13, signal=7):
    """MACD指标计算"""
    df['fast_ema'] = calculate_ema(df['close'], fast)
    df['slow_ema'] = calculate_ema(df['close'], slow)
    df['macd'] = df['fast_ema'] - df['slow_ema']
    df['signal'] = calculate_ema(df['macd'], signal)
    df['hist'] = 1.5 * (df['macd'] - df['signal'])
    df['deltaMACD'] = (df['macd'] - df['signal'])
    return df


def calculate_rsi(df, window=6):
    """计算RSI指标（网页3公式实现）"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def detect_divergence(stockQuery, symbol, df, lookback=90, bd_signal=False):
    start_date='20250210'
    date_filter = (df.index >= pd.to_datetime(start_date))
    """背离检测主逻辑"""
    # 极值计算
    df['lowest_macd'] = df['macd'].rolling(lookback).min()
    df['lowest_price'] = df['close'].rolling(lookback).min()
    df['highest_price'] = df['high'].rolling(lookback).max()
    df['highest_macd'] = df['macd'].rolling(lookback).max()

    # ========== 新增股性过滤层 ==========
    # 计算波动性指标
    df = calculate_volatility(df)
    # 计算大阴线频率
    df = big_red_line_filter(df)

    # 股性优质条件（网页7）
    good_character_cond = (
        (df['amplitude_ma'] > 0.025) &              # 20日平均振幅>2%
        (~is_chop_market(df)) # 排除织布机走势
        # &
        # (df['big_red_count'] < 6)                  # 60日内大阴线<6次
    )

    # ========== 新增成交量缩量条件 ==========
    # 计算20日成交量中位数（网页7基准量逻辑）
    # df['vol_median_20'] = df['volume'].rolling(20, min_periods=20).median().shift(1)
    # 缩量条件：当日成交量<基准量60%（网页2动态参数优化思想）
    # volume_cond = (df['volume'] > df['vol_median_20'] * 0.6)

    # ====== 新增代码段：3日阶梯缩量验证 ======
    # 计算连续3日成交量递减（网页5阶梯递减指标原理）
    # df['vol_decrease'] = df['volume'].rolling(3, min_periods=3).apply(
    #     lambda x: (x[0] < x[1]) | (x[1] < x[2]), raw=True
    # ).shift(1)  # 避免未来函数

    # 严格模式：连续3日绝对递减（网页2动态参数优化）
    # volume_decline_cond = df['vol_decrease'] == True

    # ========== 新增RSI条件 ==========
    # df = calculate_rsi(df)  # 计算RSI指标

    roe = stockQuery.get_stock_roe(symbol)
    # 修改后（假设roe是标量）
    roe_condition = (roe is not None) & (pd.notna(roe)) & (roe >= 5)  # 标量处理

    # 底背离条件
    bottom_cond = (
            (date_filter) &  # 新增日期条件[3](@ref)
            (df['close'] <= df['lowest_price'] * 1.01) &  # 价格接近周期低点
            (df['macd'] >= df['lowest_macd'] * 1.1)
            &  # MACD高于周期低点110%
            (df['above_30week'])  # 新增均线过滤
            &  # 布林带宽度
            (df['boll_width'] >= 0.25)
            # &  # 缩量条件
            # (volume_cond)
            # &  # 新增阶梯缩量
            # (volume_decline_cond)
            # &  # 超卖区域
            # (df['RSI'] < 30)
            &  # roe
            roe_condition
            &  # 20日平均振幅>2%，排除织布机走势
            (df['amplitude_ma'] > 0.02)
    )
    df['预底'] = np.where(bottom_cond, df['macd'], np.nan)
    if bd_signal:
        return df[['预底']].dropna(how='all')
    else:
        top_cond = (
                (df['close'] >= df['highest_price'] * 0.99) &  # 价格接近周期高点
                (df['macd'] <= df['highest_macd'] * 0.9) &  # MACD低于周期高点90%
                (df['above_30week'])  # 新增均线过滤
        )
        # 顶背离条件
        df['预顶'] = np.where(top_cond, df['macd'], np.nan)
        return df[['预顶', '预底']].dropna(how='all')


if __name__ == '__main__':
    # 参数设置
    symbol = 'sh600563'  # 平安银行
    start_date = '20240501'
    hold_periods = [1, 3, 5, 10]  # 需计算的持有周期

    # 获取数据
    df, is_cached = get_stock_data(symbol, start_date)

    # 添加均线计算
    df = calculate_moving_averages(df)
    # 计算MACD
    macd_df = calculate_macd(df)
    # 新增收益预计算
    df = calculate_returns(macd_df, hold_periods)
    query_tool = stockCsv.StockQuery()
    # 检测背离
    signals = detect_divergence(query_tool,symbol, macd_df, 60, True)

    # 格式化输出
    print(f"\n背离信号收益分析报告：{query_tool.get_name_by_code(symbol)}")
    for date, row in signals.iterrows():
        if not pd.isna(row['预底']):
            buy_price = df.loc[date, 'close']
            report = f"{date.strftime('%Y-%m-%d')} | 买入价：{buy_price:.2f}"

            # 遍历所有持有周期
            for days in hold_periods:
                ret = df.loc[date, f'return_{days}d']
                if pd.isna(ret):
                    report += f" | {days}天：数据不足"
                else:
                    report += f" | {days}天：{ret:.2%}"

            print(report)
