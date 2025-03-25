import time
from datetime import datetime

import akshare as ak
import pandas as pd

# 配置参数
SYMBOL = "603881"  # 股票代码
BOLL_WINDOW = 20  # BOLL计算周期
STD_DEV = 2  # 标准差倍数
ALERT_THRESHOLD = 0.005  # 触轨阈值(0.5%)
CHECK_INTERVAL = 10  # 检查间隔(秒)


def fetch_intraday_data():
    """获取并预处理分时数据
       拿到的是一个DataFrame 对象 二列，多行
    """
    try:
        df = ak.stock_intraday_em(symbol=SYMBOL)
        if df.empty:
            raise ValueError("返回数据为空")
        # print("时间："+df['时间'])
        # 字段标准化，屁用没有，转换为 datetime 类型：使用 pd.to_datetime 将拼接后的字符串转换为 Pandas 的 datetime 类型，便于后续时间操作。
        df['时间'] = pd.to_datetime(datetime.now().strftime("%Y-%m-%d ") + df['时间'])
        # 这里拿到的时间是list，直接打印伤不起
        # 将 DataFrame 中的列名从 成交价 改为 close，从 手数 改为 volume。nplace=True 表示直接在原 DataFrame 上修改，而非返回一个新的 DataFrame。
        df.rename(columns={'成交价': 'close', '手数': 'volume'}, inplace=True)
        # 将 时间 列设为 DataFrame 的索引（行标签），按时间顺序对 DataFrame 进行排序（也就是时间）。仅保留 close 和 volume 两列。
        return df.set_index('时间').sort_index()[['close', 'volume']]
    except Exception as e:
        print(f"数据获取失败: {str(e)}")
        return pd.DataFrame()


def resample_5min(raw_df):
    """生成5分钟OHLC数据（修正版）"""
    if raw_df.empty:
        return raw_df

    # 动态扩展OHLC列
    # 将 close 列按 5 分钟周期重采样。在每个 5 分钟周期内，取第一个 close 值作为 open（开盘价）。将计算结果存储到 raw_df 的新列 open 中。
    raw_df['open'] = raw_df['close'].resample('5min').transform('first')
    raw_df['high'] = raw_df['close'].resample('5min').transform('max')
    raw_df['low'] = raw_df['close'].resample('5min').transform('min')

    # 重采样聚合
    resampled = raw_df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # 从 resampled 中筛选出 volume 大于 0 的行。
    return resampled[resampled['volume'] > 0]


def calculate_boll(df):
    """计算BOLL指标"""
    if len(df) < BOLL_WINDOW:
        return df

    # 计算 close 列的 20 周期移动平均线（中轨）。
    df['MA20'] = df['close'].rolling(BOLL_WINDOW).mean()
    # 计算 close 列的 20 周期标准差。
    std = df['close'].rolling(BOLL_WINDOW).std()
    # 计算布林带上轨（中轨 + 2 倍标准差），并保留两位小数。
    df['Upper'] = (df['MA20'] + STD_DEV * std).round(2)  # 保留两位小数
    # 计算布林带下轨（中轨 - 2 倍标准差），并保留两位小数。
    df['Lower'] = (df['MA20'] - STD_DEV * std).round(2)
    return df.dropna()


def check_alert(current_price, upper, lower):
    """触轨检测"""
    if current_price >= upper * (1 - ALERT_THRESHOLD):
        return f"⚠️ 触及上轨 | 当前价：{current_price:.2f} | 上轨：{upper:.2f}"
    elif current_price <= lower * (1 + ALERT_THRESHOLD):
        return f"⚠️ 触及下轨 | 当前价：{current_price:.2f} | 下轨：{lower:.2f}"
    return f"未触及 | 当前价：{current_price:.2f} | 上轨：{upper:.2f} | 下轨：{lower:.2f}"


def trading_time_check():
    """交易时段验证"""
    now = datetime.now().time()
    morning = (time(9, 30), time(11, 30))
    afternoon = (time(13, 0), time(15, 0))
    return (morning[0] <= now <= morning[1]) or (afternoon[0] <= now <= afternoon[1])


def main_loop():
    """主监控循环"""
    historical_data = pd.DataFrame()

    while True:
        # if not trading_time_check():
        #     print("当前为非交易时段，暂停监控")
        #     time.sleep(3600)
        #     continue

        try:
            # 获取并处理数据
            raw_data = fetch_intraday_data()
            if raw_data.empty:
                time.sleep(60)
                continue

            # 生成5分钟K线
            new_5m_data = resample_5min(raw_data)

            # 合并历史数据（去重处理）
            # 将新数据（new_5m_data）与历史数据（historical_data）合并，并去除重复行。
            # 仅合并新数据中时间戳大于历史数据最后一条记录的部分，避免重复处理。
            # 实际上没有必要，因为得到的数据都是最全的当日，哦。这是处理好的5分钟数据，还是要反复更新的。
            # 可能是这段代码，导致的最后五分钟k线一直不更新
            # if not historical_data.empty:
            #     new_5m_data = new_5m_data[new_5m_data.index > historical_data.index[-1]]
            # historical_data = pd.concat([historical_data, new_5m_data]).drop_duplicates()

            # 计算BOLL指标
            if len(historical_data) >= BOLL_WINDOW:
                boll_data = calculate_boll(historical_data.copy())
                if not boll_data.empty:
                    # 从 boll_data 中提取最后一行数据（即最新时间戳对应的数据）。
                    latest = boll_data.iloc[-1]

                    # 触发警报检测
                    alert_msg = check_alert(latest['close'], latest['Upper'], latest['Lower'])
                    if "⚠️" in str(alert_msg):
                        print("\n" + "=" * 40)
                        print(f"【{SYMBOL}】{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(alert_msg)
                        print("最新3根K线：")
                        print(boll_data[['close', 'Upper', 'Lower']].tail(3))
                        print("=" * 40 + "\n")
                    else:
                        print(f"【{SYMBOL}】{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(alert_msg)
            # 等待下次检查
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print(f"监控异常：{str(e)}")
            time.sleep(60)


if __name__ == "__main__":
    print(f"启动{SYMBOL}股票5分钟BOLL监控...")
    main_loop()
