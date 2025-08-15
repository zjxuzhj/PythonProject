import os
import time

import akshare as ak
import pandas as pd
from fastparquet import write
from fastparquet import ParquetFile

import getAllStockCsv
import getTopIndustry as getTopIndustry
import getBacktestForDepart as backtestForDepart
import scanAllStockDepart as scan
import http_util

def addStockData(symbol, start_date):
    file_name = f"stock_{symbol}_{start_date}.parquet"
    cache_path = os.path.join("data_cache", file_name)


def get_stock_prefix(code):
    code_str = str(code).zfill(6)  # 确保6位字符串，如输入"688131"→"688131"
    prefix = ""

    # 判断逻辑
    if code_str.startswith(("688","60")):
        prefix = "sh"
    elif code_str.startswith(("00","30")):
        prefix = "sz"
    elif code_str.startswith(("430", "8", "9")):
        prefix = "bj"
    else:
        prefix = "未知"

    return f"{prefix}{code_str}"


query_tool = getAllStockCsv.StockQuery()

# 使用示例（单独执行此类时）
if __name__ == "__main__":

    # 获取实时数据
    spot_df = ak.stock_zh_a_spot_em()  # 东财接口更稳定[6](@ref)
    # 字段映射表（中文→英文）
    column_mapping = {
        '代码': 'symbol',
        '最新价': 'close',
        '今开': 'open',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'amount',
        '涨跌幅': 'pct_chg',
        '涨跌额': 'change',
        '换手率': 'turnover_rate'
    }

    # 遍历处理每只股票（网页3的遍历方法）
    for idx, row in spot_df.iterrows():
        try:
            # 提取标准化代码（处理带交易所前缀的情况）
            symbol = row['代码'].split('.')[0]  # 如"600519.SH"→"600519"

            file_name = f"stock_{get_stock_prefix(symbol)}_20240201.parquet"
            cache_path = os.path.join("data_cache", file_name)

            # 读取数据并保留索引
            try:
                df = pd.read_parquet(cache_path, engine='fastparquet')
                # 若原数据没有date索引，则创建索引
                if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                    df = df.set_index('date')
            except FileNotFoundError:
                df = pd.DataFrame()
                continue  # 直接跳过当前循环的后续代码

            today = time.strftime("%Y-%m-%d", time.localtime())
            # 新数据准备（带日期索引）
            new_date = pd.to_datetime(today)
            # 构建新数据字典
            new_data = {
                "股票代码": symbol,
                "open": row['今开'],
                "close": row['最新价'],
                "high": row['最高'],
                "low": row['最低'],
                "volume": row['成交量'],
                "成交额": row['成交额'],
                "振幅": row['振幅'],
                "涨跌幅": row['涨跌幅'],
                "涨跌额": row['涨跌额'],
                "换手率": row['换手率']
            }

            # 索引存在性检查
            if not df.empty and new_date in df.index:
                # 更新模式（保留原始数据类型）
                df.loc[new_date, list(new_data.keys())] = list(new_data.values())
                print(f"已更新 {symbol} {new_date} 数据")
            else:
                # 追加模式（带索引创建）
                new_row = pd.DataFrame([new_data], index=[new_date])
                df = pd.concat([df, new_row])
                print(f"已新增 {symbol} {new_date} 数据")

            # 保存时保持索引
            df.to_parquet(
                cache_path,
                engine='fastparquet',
                compression='snappy'
            )

        except Exception as e:
            print(f"处理 {row['代码']} 失败：{str(e)}")

    print("全市场数据更新完成")

    http_util.updateZTThemeAndTime()
    print("今日涨停原因以及时间更新完毕")

    time.sleep(60) # 防止短时间多次调用被屏蔽
    query_tool.update_stock_market_value()
    print("--------------- 今日股票市值更新完毕 ---------------")
    # 记住每天盘后更新数据，不然会对卖出逻辑产生影响
    print("--------------- 所有数据均更新完毕，请放心！！！---------------")

    # getTopIndustry.get_top_industry()
    #
    # scan.setup_logger()
    #
    # # 记录总耗时起点
    # total_start = time.perf_counter()
    #
    # # 加载股票列表并过滤
    # filtered_stocks = query_tool.get_all_filter_stocks()
    #
    # # 分批处理
    # result_df = scan.batch_process(filtered_stocks[['stock_code', 'stock_name']].values)
    #
    # # Excel格式输出部分
    # excel_start = time.perf_counter()
    #
    # # 格式化输出
    # writer = pd.ExcelWriter('signals.xlsx', engine='xlsxwriter')
    # result_df.to_excel(writer, index=False, sheet_name='背离信号')
    #
    # # 设置Excel格式
    # workbook = writer.book
    # format_red = workbook.add_format({'font_color': '#FF0000'})
    # format_green = workbook.add_format({'font_color': '#00B050'})
    #
    # worksheet = writer.sheets['背离信号']
    # worksheet.conditional_format('D2:D1000', {
    #     'type': 'text',
    #     'criteria': 'containing',
    #     'value': '顶',
    #     'format': format_red
    # })
    # worksheet.conditional_format('D2:D1000', {
    #     'type': 'text',
    #     'criteria': 'containing',
    #     'value': '底',
    #     'format': format_green
    # })
    # writer.close()
    # excel_duration = time.perf_counter() - excel_start
    #
    # # 计算总耗时
    # total_duration = time.perf_counter() - total_start
    #
    # # 输出耗时统计（带人性化格式）
    # scan.logger.info("\n" + "=" * 50)
    # scan.logger.info(f"Excel格式处理耗时: {excel_duration:.2f}s")
    # scan.logger.info(
    #     f"总耗时: {total_duration // 3600:.0f}h {(total_duration % 3600) // 60:.0f}m {total_duration % 60:.2f}s")
    # scan.logger.info("=" * 50)
    #
    # time.sleep(3)
    # backtestForDepart.batch_process(
    #     input_path="signals.xlsx",  # 输入文件路径
    #     output_path="output_20250329.xlsx"  # 带日期的输出文件名
    # )

