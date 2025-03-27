import akshare as ak
import numpy as np
import pandas as pd

import getAllStockCsv as stockCsv


def get_stock_data(symbol, start_date):
    if len(symbol)>6:
        symbol = stockCsv.StockQuery().get_simple_by_code(symbol)

    """获取日线数据（复权处理）"""
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        adjust="qfq"  # 前复权
    )
    df.rename(columns={
        '日期': 'date',
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '收盘': 'close',
        '成交量': 'volume'
    }, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date').sort_index()

def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def calculate_macd(df, fast=5, slow=13, signal=7):
    """MACD指标计算"""
    df['fast_ema'] = calculate_ema(df['close'], fast)
    df['slow_ema'] = calculate_ema(df['close'], slow)
    df['macd'] = df['fast_ema'] - df['slow_ema']
    df['signal'] = calculate_ema(df['macd'], signal)
    df['hist'] = 1.5 * (df['macd'] - df['signal'])
    df['deltaMACD'] = (df['macd'] - df['signal'])
    return df


def detect_divergence(df, lookback=90):
    """背离检测主逻辑"""
    # 极值计算
    df['lowest_macd'] = df['macd'].rolling(lookback).min()
    df['lowest_price'] = df['close'].rolling(lookback).min()
    df['highest_price'] = df['high'].rolling(lookback).max()
    df['highest_macd'] = df['macd'].rolling(lookback).max()

    # 顶背离条件
    top_cond = (
            (df['close'] >= df['highest_price'] * 0.99) &  # 价格接近周期高点
            (df['macd'] <= df['highest_macd'] * 0.9)  # MACD低于周期高点90%
    )

    # 底背离条件
    bottom_cond = (
            (df['close'] <= df['lowest_price'] * 1.01) &  # 价格接近周期低点
            (df['macd'] >= df['lowest_macd'] * 1.1)  # MACD高于周期低点110%
    )

    # 标记信号
    df['预顶'] = np.where(top_cond, df['macd'], np.nan)
    df['预底'] = np.where(bottom_cond, df['macd'], np.nan)

    return df[['预顶', '预底']].dropna(how='all')


if __name__ == '__main__':
    # 参数设置
    symbol = 'sh603650'  # 平安银行
    start_date = '20241001'

    # 获取数据
    df = get_stock_data(symbol, start_date)

    # 计算MACD
    macd_df = calculate_macd(df)

    # 检测背离
    signals = detect_divergence(macd_df)

    # 格式化输出
    print("背离信号检测结果：")
    for date, row in signals.iterrows():
        if not pd.isna(row['预顶']):
            print(f"{date.strftime('%Y-%m-%d')} 预顶信号 | MACD值：{row['预顶']:.2f}")
        if not pd.isna(row['预底']):
            print(f"{date.strftime('%Y-%m-%d')} 预底信号 | MACD值：{row['预底']:.2f}")