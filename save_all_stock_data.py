import logging
import time
import os
import akshare as ak
import getAllStockCsv as stockCsv
import pandas as pd
import getAllStockCsv

query_tool = getAllStockCsv.StockQuery()
def setup_logger():
    logger = logging.getLogger('stock_analysis')
    if logger.handlers:  # 关键判断：避免重复配置
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 关闭层级传播

    # 控制台Handler（仅添加一次）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
logger = setup_logger()

# 带频率控制的批量处理
def update_data(stock_list):
    results = []

    start_date = '20190101'
    total = len(stock_list)
    logger.info(f" 开始批量处理，共{total}只股票")

    # for i in range(0, 2000, batch_size):
    for i in range(0, len(stock_list), 1):
        # for i in range(1500, 1600, batch_size):
        batch = stock_list[i:i + 1]
        for code, name in batch:
            logger.info(f"⚪ 开始处理 {code} {name}")
            # 获取数据，# 获取数据及缓存状态
            file_name = f"stock_{code}_{start_date}.parquet"
            cache_path = os.path.join("back_test_data_cache", file_name)

            # 尝试读取缓存
            if os.path.exists(cache_path):
                try:
                    df = pd.read_parquet(cache_path, engine='fastparquet')
                    print(f"从缓存加载数据：{code}")
                except Exception as e:
                    print(f"缓存读取失败：{e}（建议删除损坏文件：{cache_path}）")
                continue

            # 缓存不存在时获取新数据
            try:
                if len(code) > 6:
                    code = stockCsv.StockQuery().get_simple_by_code(code)

                """获取日线数据（复权处理）"""
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    adjust="qfq"
                )
                # 数据标准化处理
                df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume'
                }, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                # 保存到本地
                os.makedirs("back_test_data_cache", exist_ok=True)
                df.set_index('date').to_parquet(  # 保存时带索引
                    cache_path,
                    engine='fastparquet',
                    compression='snappy'
                )
                print(f"新数据已缓存：{code}")

            except Exception as e:
                print(f"数据获取失败：{code} - {str(e)}")
                return pd.DataFrame(), False

        # 根据API使用情况控制频率
        time.sleep(3)
    return pd.DataFrame(results)


if __name__ == '__main__':
    setup_logger()

    # 记录总耗时起点
    total_start = time.perf_counter()
    # 加载股票列表并过滤
    filtered_stocks = query_tool.get_all_filter_stocks()
    # 分批处理
    result_df = update_data(filtered_stocks[['stock_code', 'stock_name']].values)
    # 计算总耗时
    total_duration = time.perf_counter() - total_start
    # 输出耗时统计（带人性化格式）
    logger.info("\n" + "=" * 50)
    logger.info(
        f"总耗时: {total_duration // 3600:.0f}h {(total_duration % 3600) // 60:.0f}m {total_duration % 60:.2f}s")
    logger.info("=" * 50)
