from datetime import datetime, timedelta
import os
import requests
import re
import pandas as pd

def get_yesterday(date_str):
    # 将输入的字符串转换为日期对象
    date_obj = datetime.strptime(date_str, "%Y%m%d")

    # 计算昨天的日期
    yesterday = date_obj - timedelta(days=1)

    # 将结果格式化为YYYYMMDD
    return yesterday.strftime("%Y%m%d")

def get_yesterdayNew(date_str):
    # 将输入的字符串转换为日期对象
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    # 计算昨天的日期
    yesterday = date_obj - timedelta(days=1)

    # 将结果格式化为YYYYMMDD
    return yesterday.strftime("%Y-%m-%d")

# 假设已定义节假日列表（需根据实际情况维护或对接交易所API）
holidays = [
    "2025-01-01",  # 元旦
    "2025-02-10",  # 春节
    "2025-04-04",  # 清明节
    "2025-05-01",  # 劳动节
    "2025-10-01",  # 国庆节
    # ... 其他节假日
]

def get_previous_trading_day(today_str, exchange="SSE"):
    """
    获取指定日期的上一个交易日（跳过周末和节假日）

    Args:
        today_str (str): 当前日期，格式 "YYYY-MM-DD"
        exchange (str): 交易所标识（默认为上交所 SSE）

    Returns:
        str: 上一个交易日的日期字符串，格式 "YYYY-MM-DD"
    """
    # 转换为 datetime 对象
    current_date = datetime.strptime(today_str, "%Y-%m-%d").date()

    # 循环查找上一个交易日
    max_retry = 30  # 防止无限循环（如连续假期）
    for _ in range(max_retry):
        current_date -= timedelta(days=1)

        # 1. 跳过周末（周一=0, 周日=6）
        if current_date.weekday() in [5, 6]:  # 周六、周日
            continue

        # 2. 跳过节假日
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str in holidays:
            continue

        # 如果是交易日，返回日期
        return date_str

    # 超出最大重试仍未找到（理论上不会触发）
    raise ValueError(f"无法在 {max_retry} 天内找到上一个交易日")

# 计算增长率
def calculate_growth_rate(today_num, yesterday_num):
    if yesterday_num == 0:  # 避免除零错误
        return 0.00
    growth_rate = ((today_num - yesterday_num) / yesterday_num) * 100
    return round(growth_rate, 2)  # 保留两位小数
