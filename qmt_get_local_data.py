import xtquant.xtdata as xtdata
from datetime import datetime
import pandas as pd
import os

stock_code = "600004.SH"  # Stock code
target_date = "20250506"
target_price = 13.44  # <<< PLEASE PROVIDE THE TARGET PRICE HERE

# -----------------------------------------

print(f"查询目标: 股票 {stock_code} 在 {target_date} 首次触及价格 {target_price} 的分钟")

# 2. 从本地获取数据
# get_local_data返回一个字典，其中每个键（如'open', 'close'）对应一个DataFrame
data = xtdata.get_local_data(
    field_list=['open', 'high', 'low', 'close', 'volume', 'amount'],
    stock_list=[stock_code],
    period='1m',
    start_time="20240101",
    end_time="20261030",
)

# 3. 将返回的字典数据转换为一个统一的DataFrame
df = pd.DataFrame()
# 根据您的描述，data[stock_code]本身就是一个DataFrame
if stock_code in data and not data[stock_code].empty:
    df = data[stock_code]
    # 将格式为 YYYYMMDDHHMMSS 的数字索引转换为标准DatetimeIndex
    try:
        df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d%H%M%S')
    except Exception as e:
        print(f"转换时间索引时出错: {e}")
        df = pd.DataFrame()  # 如果转换失败，则重置df为空
else:
    print(f"错误: 未能获取到 {stock_code} 在 {target_date} 的本地数据。")

if not df.empty:
    # 4. 核心逻辑: 查找价格首次触及的分钟
    # 条件：分钟最低价 <= 目标价 <= 分钟最高价
    # 由于DataFrame已按时间排序，第一个满足条件的就是我们寻找的目标
    touched_df = df[(df['low'] <= target_price) & (df['high'] >= target_price)]

    # 5. 输出结果
    print("\n--- 查询结果 ---")
    if not touched_df.empty:
        first_touch_minute = touched_df.iloc[0]
        first_touch_time = first_touch_minute.name

        print(f"✅ 目标价格 {target_price} 在 {first_touch_time.strftime('%Y-%m-%d %H:%M:%S')} 首次触及并成交。")
        print("\n当分钟详细行情:")
        # 为了更美观的显示，我们在这里创建一个Series
        info = pd.Series({
            "时间": first_touch_time.strftime('%H:%M'),
            "开盘价": first_touch_minute['open'],
            "最高价": first_touch_minute['high'],
            "最低价": first_touch_minute['low'],
            "收盘价": first_touch_minute['close'],
            "成交量(手)": int(first_touch_minute['volume']),
            "成交额(元)": int(first_touch_minute['amount'])
        })
        print(info.to_string())
    else:
        print(f"❌ 在 {target_date} 的交易时间内，价格从未触及 {target_price}。")
        # 打印当天的价格范围以供参考
        if not df.empty:
            print(f"当日价格范围: 最低价 {df['low'].min()}, 最高价 {df['high'].max()}")

else:
    print("未能进行查询，因为处理后的数据为空或格式不正确。")

# if __name__ == '__main__':
#     print(f"下载完成: {stock_code} {start_date} 至 {end_date} 的1分钟数据")