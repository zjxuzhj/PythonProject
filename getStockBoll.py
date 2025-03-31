import datetime
import time

import akshare as ak
import pandas as pd
from serverchan_sdk import sc_send

import getAllStockCsv as stockCsv
import util
import os

# 配置参数
SYMBOL = "603881"  # 股票代码
BOLL_WINDOW = 20  # BOLL计算周期
STD_DEV = 2  # 标准差倍数
ALERT_THRESHOLD = 0.005  # 触轨阈值(0.5%)
CHECK_INTERVAL = 10  # 检查间隔(秒)

# 在代码全局区域添加
DATA_FOLDER = "stock_data"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

today_str = datetime.datetime.now().strftime("%Y-%m-%d")
query_tool = stockCsv.StockQuery()


def fetch_intraday_data(df, isOld=False):
    """获取并预处理分时数据
       拿到的是一个DataFrame 对象 二列，多行
    """
    try:
        if df.empty:
            raise ValueError("返回数据为空")
        # print("时间："+df['时间'])
        # 字段标准化，屁用没有，转换为 datetime 类型：使用 pd.to_datetime 将拼接后的字符串转换为 Pandas 的 datetime 类型，便于后续时间操作。
        if isOld:
            df['时间'] = pd.to_datetime(df['时间'])
        else:
            df['时间'] = pd.to_datetime(datetime.datetime.now().strftime("%Y-%m-%d ") + df['时间'])
        # 这里拿到的时间是list，直接打印伤不起
        # 将 DataFrame 中的列名从 成交价 改为 close，从 手数 改为 volume。nplace=True 表示直接在原 DataFrame 上修改，而非返回一个新的 DataFrame。
        if isOld:
            df.rename(columns={'收盘': 'close', '成交量': 'volume'}, inplace=True)
        else:
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
    std = df['close'].rolling(BOLL_WINDOW).std(ddof=0)  # 关键参数修正[5,6](@ref)
    df['std'] = std
    # 计算布林带上轨（中轨 + 2 倍标准差），并保留两位小数。
    df['Upper'] = (df['MA20'] + STD_DEV * std).round(2)  # 保留两位小数
    # 计算布林带下轨（中轨 - 2 倍标准差），并保留两位小数。
    df['Lower'] = (df['MA20'] - STD_DEV * std).round(2)
    return df.dropna()


def check_alert(current_price, upper, lower, std):
    """触轨检测"""
    if current_price >= upper:
        # 计算标准差
        # 0.5倍标准差：温和偏离，可能为趋势延续信号。也就是0.25
        # 1倍标准差：显著超买，需警惕回调风险。也就是0.5
        # 2倍标准差：极端波动（统计学上概率＜5%），通常预示价格回归均值。也就是1
        deviation = ((current_price - upper) / 2 * std)
        deviation_rounded = round(deviation, 1)
        return f"⚠️ 触及上轨 | 当前价：{current_price:.2f} | 上轨：{upper:.2f} | 超出：{deviation_rounded:.1f}倍"
    elif current_price <= lower:
        deviation = ((lower - current_price) / 2 * std)
        deviation_rounded = round(deviation, 1)
        return f"⚠️ 触及下轨 | 当前价：{current_price:.2f} | 下轨：{lower:.2f} | 超出：{deviation_rounded:.1f}倍"
    return f"未触及 | 当前价：{current_price:.2f} | 上轨：{upper:.2f} | 下轨：{lower:.2f}"

def get_stock_filepath(symbol):
    return f"{DATA_FOLDER}/{symbol}.csv"

def save_historical_data(symbol, df):
    """保存历史数据到CSV"""
    filepath = get_stock_filepath(symbol)
    # 追加模式写入，首次写入包含表头
    df.to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=True)
    print(f"已保存{len(df)}条数据到 {filepath}")

def load_historical_data(symbol):
    """从CSV加载历史数据"""
    filepath = get_stock_filepath(symbol)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=['时间'], index_col='时间')
        return df.sort_index()
    return pd.DataFrame()

def trading_time_check():
    now = datetime.datetime.now().time()
    morning = (datetime.time(9, 30), datetime.time(11, 30))
    afternoon = (datetime.time(13, 0), datetime.time(15, 0))
    return (morning[0] <= now <= morning[1]) or (afternoon[0] <= now <= afternoon[1])


# 在全局区域添加时间记录变量
LAST_SEND_TIME = 0  # 初始化最后发送时间戳[1](@ref)


def main_loop():
    """主监控循环"""
    global LAST_SEND_TIME  # 声明全局变量
    historical_data = pd.DataFrame()

    while True:
        if not trading_time_check():
            print("当前为非交易时段，暂停监控")
            time.sleep(300)
            continue

        try:
            # 获取并处理数据
            df = ak.stock_intraday_em(symbol=SYMBOL)
            newdf = df.copy()  # 强制生成独立副本
            newdf = newdf[newdf['时间'] >= '09:30:00']
            raw_data = fetch_intraday_data(newdf, False)
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
            historical_data = new_5m_data
            if len(historical_data) < BOLL_WINDOW:
                print("当天数据不够，合并前一天k线数据")
                yesToday_str = util.get_previous_trading_day(today_str)
                filepath = get_stock_filepath(SYMBOL)

                if os.path.exists(filepath):
                    # 从本地加载前一天的数据
                    new_5m_dayBeforeData = load_historical_data(SYMBOL)
                else:
                    # 从API获取并保存前一天的数据，延迟五秒
                    time.sleep(5)
                    dayBeforeDf = ak.stock_zh_a_hist_min_em(
                        symbol=SYMBOL,
                        start_date=f"{yesToday_str} 09:30:00",
                        end_date=f"{yesToday_str} 15:00:00",
                        period="5",
                        adjust="qfq",
                    )
                    raw_dayBeforeData = fetch_intraday_data(dayBeforeDf, True)
                    new_5m_dayBeforeData = resample_5min(raw_dayBeforeData)
                    try:
                        save_historical_data(SYMBOL, new_5m_dayBeforeData)
                    except PermissionError:
                        print(f"文件 {filepath} 被占用，保存失败")
                    except Exception as e:
                        print(f"存储异常: {str(e)}")

                # 新增合并逻辑
                historical_data = pd.concat([new_5m_dayBeforeData, historical_data]).sort_index()
                historical_data = historical_data[~historical_data.index.duplicated()]  # 去重（避免时间重叠）

            # 计算BOLL指标
            if len(historical_data) >= BOLL_WINDOW:
                boll_data = calculate_boll(historical_data.copy())
                if not boll_data.empty:
                    # 从 boll_data 中提取最后一行数据（即最新时间戳对应的数据）。
                    latest = boll_data.iloc[-1]

                    # 触发警报检测
                    alert_msg = check_alert(latest['close'], latest['Upper'], latest['Lower'], latest['std'])
                    if "⚠️" in str(alert_msg):
                        print("\n" + "=" * 40)
                        print(
                            f"【{query_tool.get_name_by_code(stockCsv.add_stock_prefix(SYMBOL))}】{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(alert_msg)
                        print("最新3根K线：")
                        print(boll_data[['close', 'Upper', 'Lower']].tail(3))
                        print("=" * 40 + "\n")

                        current_time = time.time()
                        # 频率控制逻辑（60秒间隔）
                        if current_time - LAST_SEND_TIME >= 60:
                            # 发送消息并更新最后发送时间
                            title = f"{query_tool.get_name_by_code(stockCsv.add_stock_prefix(SYMBOL))}" + alert_msg
                            response = sc_send("SCT248551TKIaBVraC3CpN1ei1cqTJJhXU", title)
                            LAST_SEND_TIME = current_time  # 更新发送时间戳[1](@ref)
                        else:
                            print(f"消息发送冷却中（剩余{60 - (current_time - LAST_SEND_TIME):.0f}秒）")
                    else:
                        print(f"【{SYMBOL}】{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(alert_msg)
            # 等待下次检查
            else:
                print("K线数量不满足绘制条件")
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print(f"监控异常：{str(e)}")
            time.sleep(60)


if __name__ == "__main__":
    print(f"启动{SYMBOL}股票5分钟BOLL监控...")
    # 初始化查询器

    # print(query_tool.get_name_by_code(stockCsv.add_stock_prefix(SYMBOL)))  # 输出代码对应名称
    # print(query_tool.get_code_by_name("数据港"))  # 输出：603881

    # today_str = "2025-03-31"  # 假设周一
    # previous_trading_day = util.get_previous_trading_day(today_str)
    # print(f"上一个交易日: {previous_trading_day}")  # 输出: 2024-04-30（跳过五一假期）
    main_loop()
