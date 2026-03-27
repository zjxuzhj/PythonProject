import os
import time
from datetime import datetime
from pathlib import Path
import argparse

import pandas as pd
import xtquant.xtdata as xtdata
import getAllStockCsv


## 该方法用来补救东财数据没办法正常拉取，使用miniqmt补全一天的数据
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


def get_project_root() -> Path:
    # 从当前脚本的绝对路径开始
    current_path = Path(__file__).resolve()

    # 遍历当前目录及其所有父目录
    for parent in [current_path] + list(current_path.parents):
        # 核心逻辑：如果在这个目录下发现了 .git 文件夹 或者 requirements.txt
        if (parent / '.git').exists() or (parent / 'requirements.txt').exists():
            return parent

    # 如果都没找到，退回到当前脚本所在目录（兜底）
    return current_path.parent

def convert_to_miniqmt_format(stock_code):
    """
    将sh600138格式转换为600138.SH格式
    """
    if stock_code.startswith('sh'):
        return stock_code[2:] + '.SH'
    elif stock_code.startswith('sz'):
        return stock_code[2:] + '.SZ'
    elif stock_code.startswith('bj'):
        return stock_code[2:] + '.BJ'
    else:
        return stock_code


def get_all_stock_codes():
    """获取所有A股股票代码列表"""
    try:
        query_tool = getAllStockCsv.StockQuery()
        # filtered_stocks = query_tool.get_all_filter_stocks()
        filtered_stocks = query_tool.get_all_stocks()

        # 获取所有股票代码
        stock_codes = []
        for _, row in filtered_stocks.iterrows():
            stock_code = row['stock_code']
            stock_codes.append(stock_code)

        print(f"获取到 {len(stock_codes)} 只股票代码")
        return stock_codes
    except Exception as e:
        print(f"获取股票代码失败: {e}")
        return []


def download_stock_data_for_date_range(stock_codes, start_date, end_date):
    """
    使用miniqmt下载指定时间段的股票数据
    """
    print(f"开始下载 {start_date} 到 {end_date} 的股票数据...")

    total_stocks = len(stock_codes)
    success_count = 0
    failed_stocks = []

    start_item_number = 1
    start_index = start_item_number - 1
    stocks_to_process = stock_codes[start_index:]

    for idx, stock_code in enumerate(stocks_to_process, start_item_number):
        try:
            miniqmt_code = convert_to_miniqmt_format(stock_code)
            # print(f"[{idx}/{total_stocks}] 下载 {stock_code} -> {miniqmt_code}")

            data = xtdata.download_history_data(
                stock_code=miniqmt_code,
                period='1d',
                start_time=start_date,
                end_time=end_date
            )

            success_count += 1
            if idx % 100 == 0:
                print(f"[{idx}/{total_stocks}] 已处理 {idx} 只股票")

        except Exception as e:
            print(f"下载 {stock_code} 失败: {e}")
            failed_stocks.append(stock_code)
            continue

    print(f"\n下载完成统计:")
    print(f"成功: {success_count}/{total_stocks}")
    print(f"失败: {len(failed_stocks)}")

    return success_count, failed_stocks


def update_parquet_files_with_miniqmt_data(start_date, end_date):
    """
    从miniqmt下载的数据中提取信息，更新parquet文件
    """
    print(f"\n开始更新parquet文件，插入 {start_date} 到 {end_date} 的数据...")

    stock_codes = get_all_stock_codes()
    if not stock_codes:
        print("无法获取股票代码列表")
        return

    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)

    success_count = 0
    failed_count = 0

    miniqmt_codes = [convert_to_miniqmt_format(code) for code in stock_codes]
    print(f"开始批量获取本地数据...")
    
    # 批量获取数据，防止多次调用导致底层崩溃
    all_data = xtdata.get_local_data(
        field_list=['open', 'high', 'low', 'close', 'volume'],
        stock_list=miniqmt_codes,
        period='1d',
        start_time=start_date,
        end_time=end_date
    )

    for idx, stock_code in enumerate(stock_codes, 1):
        try:
            miniqmt_code = convert_to_miniqmt_format(stock_code)

            if miniqmt_code not in all_data or all_data[miniqmt_code].empty:
                print(f"[{idx}/{len(stock_codes)}] {stock_code}: 无数据")
                failed_count += 1
                continue

            df_new = all_data[miniqmt_code]
            root_dir = get_project_root()
            file_name = f"stock_{stock_code}_20240201.parquet"
            cache_path = os.path.join(root_dir, cache_dir, file_name)

            try:
                df = pd.read_parquet(cache_path, engine='fastparquet')
                if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                    df = df.set_index('date')
            except FileNotFoundError:
                print(f"[{idx}/{len(stock_codes)}] {stock_code}: parquet文件不存在，跳过")
                failed_count += 1
                continue

            new_rows = []
            for date_idx, day_data in df_new.iterrows():
                target_date_obj = pd.to_datetime(str(date_idx))
                new_data = {
                    "股票代码": stock_code[2:],
                    "open": day_data['open'],
                    "close": day_data['close'],
                    "high": day_data['high'],
                    "low": day_data['low'],
                    "volume": day_data['volume'],
                    "成交额": 0,
                    "振幅": 0,
                    "涨跌幅": 0,
                    "涨跌额": 0,
                    "换手率": 0
                }
                new_rows.append(pd.DataFrame([new_data], index=[target_date_obj]))

            if new_rows:
                df_to_add = pd.concat(new_rows)
                for idx_to_add, row_to_add in df_to_add.iterrows():
                    if idx_to_add in df.index:
                        df.loc[idx_to_add, list(row_to_add.keys())] = list(row_to_add.values)
                    else:
                        df = pd.concat([df, row_to_add.to_frame().T])

                df = df.sort_index()
                # 去重，保留最新数据
                df = df[~df.index.duplicated(keep='last')]

                # 在计算前先转换为数值类型，避免object类型导致的错误
                numeric_cols = ['open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

                df['pre_close'] = df['close'].shift(1)
                df['涨跌额'] = (df['close'] - df['pre_close']).round(2)
                
                # Vectorized calculations for percentages
                df['涨跌幅'] = 0.0
                mask = (df['pre_close'].notna()) & (df['pre_close'] != 0)
                df.loc[mask, '涨跌幅'] = ((df.loc[mask, '涨跌额'] / df.loc[mask, 'pre_close']) * 100).round(2)
                
                df['振幅'] = 0.0
                df.loc[mask, '振幅'] = (((df.loc[mask, 'high'] - df.loc[mask, 'low']) / df.loc[mask, 'pre_close']) * 100).round(2)

                df['涨跌额'] = df['涨跌额'].fillna(0.0)
                df['涨跌幅'] = df['涨跌幅'].fillna(0.0)
                df['振幅'] = df['振幅'].fillna(0.0)
                df = df.drop(columns=['pre_close'])

                try:
                    df.to_parquet(cache_path, engine='fastparquet', compression='snappy')
                    print(f"[{idx}/{len(stock_codes)}] {stock_code}: 更新了 {len(new_rows)} 天的数据")
                    success_count += 1
                except Exception as e:
                    print(f"[{idx}/{len(stock_codes)}] {stock_code}: 保存parquet失败 - {e}")
                    failed_count += 1
            else:
                failed_count += 1

        except Exception as e:
            import traceback
            print(f"[{idx}/{len(stock_codes)}] {stock_code}: 处理失败 - {e}")
            traceback.print_exc()
            failed_count += 1
            continue

    print(f"\nparquet文件更新完成:")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="下载并补充股票数据")
    parser.add_argument("--start_date", type=str, default="20260316", help="开始日期 YYYYMMDD")
    parser.add_argument("--end_date", type=str, default="20260319", help="结束日期 YYYYMMDD")
    parser.add_argument("--download_only", action="store_true", help="只下载不更新parquet")
    parser.add_argument("--update_only", action="store_true", help="只更新parquet不下载")
    
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date

    print("=" * 50)
    print(f"股票数据下载和更新工具: {start_date} - {end_date}")
    print("=" * 50)

    if not args.update_only:
        print("\n步骤1: 获取股票代码列表...")
        stock_codes = get_all_stock_codes()
        if not stock_codes:
            print("无法获取股票代码，程序退出")
            return

        print(f"\n步骤2: 下载 {start_date} 到 {end_date} 数据...")
        download_stock_data_for_date_range(stock_codes, start_date, end_date)

    if not args.download_only:
        print(f"\n步骤3: 更新parquet缓存文件...")
        update_parquet_files_with_miniqmt_data(start_date, end_date)

    print("\n" + "=" * 50)
    print("数据下载和更新任务完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()
