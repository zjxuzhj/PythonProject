import os
import time
from datetime import datetime

import akshare as ak
import pandas as pd

import getAllStockCsv
from data_ingestion import http_util

# ----------------- 全局对象初始化 -----------------
# 这些对象在模块加载时创建，可以被后续的所有函数共享
try:
    query_tool = getAllStockCsv.StockQuery()
except Exception as e:
    print(f"初始化 StockQuery 失败: {e}")
    query_tool = None


# ----------------- 核心功能函数 -----------------

def get_stock_prefix(code):
    """根据股票代码判断交易所前缀 (sh, sz, bj)"""
    code_str = str(code).zfill(6)
    if code_str.startswith(("688", "60")):
        return f"sh{code_str}"
    elif code_str.startswith(("00", "30")):
        return f"sz{code_str}"
    elif code_str.startswith(("430", "8", "9")):
        return f"bj{code_str}"
    else:
        return f"unknown{code_str}"


def compute_up_down_counts(spot_df: pd.DataFrame) -> tuple[int, int]:
    ret_col = None
    for c in ["涨跌幅", "涨跌率", "pct_chg", "pct_change", "returns", "change_pct", "return", "ret", "chg_pct"]:
        if c in spot_df.columns:
            ret_col = c
            break
    if ret_col is None:
        raise RuntimeError("未找到涨跌幅列")
    ser = spot_df[ret_col].astype(str).str.replace('%', '', regex=False)
    ser = pd.to_numeric(ser, errors='coerce')
    ser = ser.dropna()
    ser = ser.where(ser.abs() <= 1, ser / 100.0)
    up_count = int((ser > 0).sum())
    down_count = int((ser <= 0).sum())
    return up_count, down_count


def update_all_daily_data():
    """
    【核心公共函数】
    执行所有每日数据的更新操作。
    包括：获取最新行情、更新个股Parquet文件、更新涨停原因、更新股票市值。
    """
    print("=" * 30)
    print(f"开始执行每日数据更新任务 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 30)

    use_sina = True
    # 1. 获取全市场实时行情数据
    try:
        print("步骤 1/4: 获取全市场实时行情...")
        if use_sina:
            spot_df = ak.stock_zh_a_spot()
        else:
            spot_df = ak.stock_zh_a_spot_em()
        if spot_df.empty:
            print("错误：未能从akshare获取到实时行情数据，任务终止。")
            return
        print("实时行情获取成功。")
    except Exception as e:
        print(f"获取akshare实时行情失败: {e}")
        return

    try:
        print("\n步骤 1.1: 统计当日涨跌家数...")
        up_count, down_count = compute_up_down_counts(spot_df)
        print(f"当日涨跌家数统计完成：上涨={up_count} 下跌={down_count}")
    except Exception as e:
        print(f"统计涨跌家数失败: {e}")
        up_count = 0
        down_count = 0

    # 2. 遍历行情，更新本地Parquet数据文件
    print("\n步骤 2/4: 更新本地个股Parquet数据文件...")
    total = len(spot_df)
    for idx, row in spot_df.iterrows():
        try:
            symbol = str(row['代码'])

            # 确保文件夹存在
            cache_dir = "data_cache"
            os.makedirs(cache_dir, exist_ok=True)

            if len(symbol) > 7:
                file_name = f"stock_{symbol}_20240201.parquet"
            else:
                file_name = f"stock_{get_stock_prefix(symbol)}_20240201.parquet"
            cache_path = os.path.join(cache_dir, file_name)

            # 1. 在读取前，先检查文件是否存在
            if not os.path.exists(cache_path):
                # 2. 如果文件不存在，跳过此股票的后续所有处理
                # print(f"文件 {cache_path} 不存在，跳过。") # (可选：取消注释以查看跳过了哪些)
                continue

            try:
                df = pd.read_parquet(cache_path, engine='fastparquet')
                if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                    df = df.set_index('date')
            except FileNotFoundError:
                # 如果文件不存在，创建一个空的DataFrame，后续逻辑会新增行
                df = pd.DataFrame()

            today_str = datetime.now().strftime("%Y-%m-%d")
            new_date = pd.to_datetime(today_str)

            if use_sina:
                high = row['最高']
                low = row['最低']
                current_price = row['最新价']
                change_amount = row['涨跌额']  # 涨跌额
                yesterday_close = current_price - change_amount
                amplitude_pct = 0.0
                if yesterday_close > 0:
                    amplitude_pct = round(((high - low) / yesterday_close) * 100, 2)
                new_data = {
                    "股票代码": getAllStockCsv.get_numeric_code(symbol),
                    "open": row['今开'],
                    "close": current_price,
                    "high": high,
                    "low": low,
                    "volume": row['成交量'] / 100,
                    "成交额": row['成交额'],
                    "涨跌幅": row['涨跌幅'],
                    "涨跌额": change_amount,
                    "振幅": amplitude_pct
                }

            else:
                new_data = {
                    "股票代码": symbol, "open": row['今开'], "close": row['最新价'],
                    "high": row['最高'], "low": row['最低'], "volume": row['成交量'],
                    "成交额": row['成交额'], "振幅": row['振幅'], "涨跌幅": row['涨跌幅'],
                    "涨跌额": row['涨跌额'], "换手率": row['换手率']
                }

            if not df.empty and new_date in df.index:
                df.loc[new_date, list(new_data.keys())] = list(new_data.values())
                # print(f"  - 已更新 {symbol} 数据") # 打印信息过多，可注释掉
            else:
                new_row = pd.DataFrame([new_data], index=[new_date])
                df = pd.concat([df, new_row])
                # print(f"  - 已新增 {symbol} 数据") # 打印信息过多，可注释掉

            df.to_parquet(cache_path, engine='fastparquet', compression='snappy')

        except Exception as e:
            print(f"处理 {row['代码']} 失败：{e}")
    print(f"个股Parquet文件更新完成，共处理 {total} 条。")

    # 3. 更新涨停原因及时间
    try:
        print("\n步骤 3/4: 更新今日涨停原因及时间...")
        http_util.updateZTThemeAndTime()
        print("涨停数据更新完毕。")
    except Exception as e:
        print(f"更新涨停数据失败: {e}")

    # 4. 更新股票市值
    if not use_sina:
        if query_tool:
            try:
                print("\n步骤 4/4: 更新股票市值...")
                time.sleep(1)
                query_tool.update_stock_market_value(spot_df)
                print("股票市值更新完毕。")
            except Exception as e:
                print(f"更新股票市值失败: {e}")

    try:
        print("\n步骤 4/4:同步写入当日市场统计...")
        from daily_market_stats import update_counts_for_date
        data_cache_dir = os.path.abspath("data_cache")
        target_date = datetime.now().date()
        update_counts_for_date(data_cache_dir, target_date, up_count, down_count, datetime.now())
        print("市场统计写入完成。")
    except Exception as e:
        print(f"写入市场统计失败: {e}")

    print("\n" + "=" * 30)
    print(f"所有数据均更新完毕 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 30 + "\n")


# ----------------- 脚本执行入口 -----------------

if __name__ == "__main__":
    # --- 方案一：直接运行一次 ---
    update_all_daily_data()

    # --- 方案二：设置为定时任务，每天下午4:30执行 ---
    # print("定时数据更新服务已启动。")
    # print("任务将在每个交易日下午16:30自动执行。")
    #
    # scheduler = BackgroundScheduler(timezone="Asia/Shanghai")
    #
    # scheduler.add_job(
    #     update_all_daily_data,
    #     trigger=CronTrigger(
    #         hour=16,
    #         minute=30,
    #         day_of_week='mon-fri'  # 周一至周五执行
    #     ),
    #     id='daily_data_update_job'
    # )
    #
    # scheduler.start()
    #
    # # 为了演示，可以先手动执行一次
    # print("为了确保数据最新，启动时立即执行一次更新任务...")
    # update_all_daily_data()
    #
    # try:
    #     # 保持主线程运行，以便调度器在后台执行
    #     print("\n服务正在后台运行，请勿关闭此窗口。按 Ctrl+C 退出。")
    #     while True:
    #         time.sleep(1)
    # except (KeyboardInterrupt, SystemExit):
    #     scheduler.shutdown()
    #     print("服务已停止。")