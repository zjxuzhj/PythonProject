from datetime import datetime, timedelta


def get_yesterday(date_str):
    # 将输入的字符串转换为日期对象
    date_obj = datetime.strptime(date_str, "%Y%m%d")

    # 计算昨天的日期
    yesterday = date_obj - timedelta(days=1)

    # 将结果格式化为YYYYMMDD
    return yesterday.strftime("%Y%m%d")

# 计算增长率
def calculate_growth_rate(today_num, yesterday_num):
    if yesterday_num == 0:  # 避免除零错误
        return 0.00
    growth_rate = ((today_num - yesterday_num) / yesterday_num) * 100
    return round(growth_rate, 2)  # 保留两位小数

# 示例
date = "20250321"
yesterday = get_yesterday(date)
print(f"输入日期: {date}, 昨天日期: {yesterday}")