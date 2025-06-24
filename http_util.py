import requests
import json
import getAllStockCsv

# 1. 发送POST请求获取数据
url = "https://duanxianxia.cn/api/getZtliveData"
payload = {
    "type": "plate",  # 动态参数：复盘类型（如'plate'表示按板块）
    "date": "2025-06-21",  # 动态参数：日期（格式为YYYY-MM-DD）
    "from": "web"  # 固定参数，标识请求来源
}
query_tool = getAllStockCsv.StockQuery()
response = requests.post(url, data=payload)

if response.status_code == 200:
    # 2. 解析JSON响应
    data = response.json()

    if data.get("result") == "success":
        # 4. 转换为Python对象
        stock_data = {
            "result": data["result"],
            "url": data["url"],
            "stocks": data["list"],
        }

        # 5. 使用数据（示例：打印前5个股票）
        print("解析成功！股票数量：", len(stock_data["stocks"]))
        print("前5个股票信息：")
        for stock in stock_data["stocks"]:
            print(f"代码：{stock['code']} | 名称：{stock['name']} | 原因：{stock['ztyy']} | 时间：{stock['time']} | 状态：{stock['zt']}")
            # query_tool.add_time(getAllStockCsv.code_add_prefix(stock['code']),stock['time'])
            query_tool.add_theme(getAllStockCsv.code_add_prefix(stock['code']),stock['ztyy'])


        # 6. 完整数据存储在stock_data对象中
        # 后续可直接使用stock_data对象进行处理
    else:
        print("接口返回失败：", data)
else:
    print(f"请求失败，状态码：{response.status_code}")