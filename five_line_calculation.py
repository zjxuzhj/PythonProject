def calculate_five_day_ma(prev_prices, fifth_day_increase_percent):
    """
    计算五日移动平均线（5日均线）

    参数：
    prev_prices -- 前4个交易日的收盘价列表（顺序：[day1, day2, day3, day4]）
    fifth_day_increase_percent -- 第五个交易日相对于第四交易日的涨幅百分比（正数表示上涨，负数表示下跌）

    返回：
    第五个交易日的五日移动平均线值
    """
    # 1. 计算第五个交易日的收盘价
    day5_close = prev_prices[-1] * (1 + fifth_day_increase_percent / 100)

    # 2. 获取完整的5个交易日收盘价
    five_day_closes = prev_prices + [day5_close]

    # 3. 计算五日移动平均线（MA5）
    five_day_ma = sum(five_day_closes) / 5

    return five_day_ma


# 示例用法
if __name__ == "__main__":
    # 示例数据：前4个交易日的收盘价
    previous_prices = [5.20, 5.33,5.86, 5.62]  # 顺序：[day1, day2, day3, day4]

    # 第五个交易日涨幅（+3.5%）
    fifth_day_increase = 6

    # 计算五日线
    ma5 = calculate_five_day_ma(previous_prices, fifth_day_increase)

    print(f"第五个交易日的五日移动平均线(MA5)值为：{ma5:.2f}")