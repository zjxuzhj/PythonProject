import time

import xtquant.xtdata as xtdata

import getAllStockCsv

query_tool = getAllStockCsv.StockQuery()
# 扩展数据需要下载的步骤 (注意，1m和5m的数据只有近一年的)
# 1.现在日线级别的沪深300的数据，
# 数据覆盖 2018-01-30至2025-11-04
# 2.下载分钟级别所有股票数据3010个，用于确认日内何时买入。
# 分钟数据 2024-7-4至2025-10-31
# 3.下载日线级别所有股票数据3010个，用于模拟计算五日线价格
# 日线数据 2024-7-4至2025-10-31


# 设置股票代码和时间范围
# stock_code = "000300.SH"
# start_date = "20200101"
# end_date = "20240801"
#
# # 下载历史行情数据
# xtdata.download_history_data2(
#     stock_list=[stock_code],
#     period="1d",           # 1分钟周期
#     start_time=start_date,
#     end_time=end_date,
#     callback=lambda data: print(f"进度: {data['finished']}/{data['total']} {data['stockcode']}") ,  # 进度回调
#     incrementally=False    # 全量下载(非增量)
# )
#
# print(f"下载完成: {stock_code} {start_date} 至 {end_date} 的1分钟数据")

# 1. 设置时间范围
start_date = "20251031"
end_date = "20251109"

# 2. 设置数据周期
# 注意：迅投支持的周期包括 1d, 1m, 5m, 15m, 30m, 60m 等
period = "1m"

if __name__ == '__main__':
    filtered_stocks = query_tool.get_all_filter_stocks()
    stock_list = filtered_stocks[['stock_code', 'stock_name']].values
    total_stocks = len(stock_list)
    print(f"准备下载 {total_stocks} 只股票的 '{period}' 周期数据...")
    print(f"时间范围: {start_date} -> {end_date}")
    print("-" * 30)

    start_index = 900
    total_stocks = len(stock_list)

    # for idx, (stock_code, stock_name) in enumerate(stock_list[start  _index:], start_index + 1):
    for idx, (stock_code, stock_name) in enumerate(stock_list, 1):
        standard_code = getAllStockCsv.convert_to_standard_format(stock_code)
        print(f"[{idx}/{total_stocks}] 开始处理: {standard_code} ({stock_name})")
        # 使用增量下载模式 (incrementally=True)，SDK 会自动检查本地数据，只下载需要补充的部分，这就是最推荐的“断点续传”方式
        try:
            xtdata.download_history_data2(
                stock_list=[standard_code],
                period=period,
                start_time=start_date,
                end_time=end_date,
                incrementally=True,
                callback=lambda data: print(f"  -> 进度: {data['finished']}/{data['total']} {data['stockcode']}")
            )
            print(f"[{idx}/{total_stocks}] 完成处理: {standard_code}")

        except Exception as e:
            # 捕获可能的异常，例如网络问题或API限制
            print(f"[错误] 处理 {standard_code} 时发生异常: {e}")
            # 你可以在这里加入重试逻辑或跳过
            continue

        # 建议在每次请求之间加入短暂延时，避免请求过于频繁
        time.sleep(0.2)  # 延时0.5秒

    print("-" * 30)
    print("所有股票数据处理完毕！")