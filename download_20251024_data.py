import os
import time
from datetime import datetime
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
        filtered_stocks = query_tool.get_all_filter_stocks()
        
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

def download_stock_data_for_date(stock_codes, target_date="20251024"):
    """
    使用miniqmt下载指定日期的股票数据
    """
    print(f"开始下载 {target_date} 的股票数据...")
    
    total_stocks = len(stock_codes)
    success_count = 0
    failed_stocks = []

    # 假设你想从第 2230 个股票开始下载
    start_item_number = 1680

    # Python 索引从 0 开始，所以第 2230 个元素的索引是 2229
    start_index = start_item_number - 1

    # 1. 使用切片 [start_index:] 来获取一个从第 2230 个元素开始的新列表
    stocks_to_process = stock_codes[start_index:]

    # 2. 遍历这个新列表，并让 idx 从 2230 开始计数
    # 这样 idx 就能和你的总进度对应上
    for idx, stock_code in enumerate(stocks_to_process, start_item_number):
        try:
            # 转换为miniqmt格式
            miniqmt_code = convert_to_miniqmt_format(stock_code)
            
            print(f"[{idx}/{total_stocks}] 下载 {stock_code} -> {miniqmt_code}")
            
            # 下载当天的日线数据，会自动更新到本地数据文件
            data = xtdata.download_history_data2(
                stock_list=[miniqmt_code],
                period='1d',
                start_time="20251023",
                end_time="20251028",
                incrementally=True,
                callback=lambda data: print(f"  -> 进度: {data['finished']}/{data['total']} {data['stockcode']}")
            )
            
            success_count += 1
            
            # 每100只股票休息一下
            if idx % 50 == 0:
                print(f"已处理 {idx} 只股票，休息2秒...")
                time.sleep(3)
            else:
                time.sleep(0.1)  # 短暂延时避免请求过频
                
        except Exception as e:
            print(f"下载 {stock_code} 失败: {e}")
            failed_stocks.append(stock_code)
            continue
    
    print(f"\n下载完成统计:")
    print(f"成功: {success_count}/{total_stocks}")
    print(f"失败: {len(failed_stocks)}")
    
    if failed_stocks:
        print(f"失败的股票: {failed_stocks[:10]}...")  # 只显示前10个
    
    return success_count, failed_stocks

def update_parquet_files_with_miniqmt_data(target_date="20251024"):
    """
    从miniqmt下载的数据中提取信息，更新parquet文件
    """
    print(f"\n开始更新parquet文件，插入 {target_date} 的数据...")
    
    # 获取所有股票代码
    stock_codes = get_all_stock_codes()
    if not stock_codes:
        print("无法获取股票代码列表")
        return
    
    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    success_count = 0
    failed_count = 0
    
    for idx, stock_code in enumerate(stock_codes, 1):
        try:
            # 转换为miniqmt格式
            miniqmt_code = convert_to_miniqmt_format(stock_code)
            
            # 从miniqmt获取数据
            data = xtdata.get_local_data(
                field_list=['open', 'high', 'low', 'close', 'volume'],
                stock_list=[miniqmt_code],
                period='1d',
                start_time=target_date,
                end_time=target_date
            )
            
            if miniqmt_code not in data or data[miniqmt_code].empty:
                print(f"[{idx}/{len(stock_codes)}] {stock_code}: 无数据")
                failed_count += 1
                continue
            
            # 获取当天数据
            day_data = data[miniqmt_code].iloc[0]
            
            # 构建parquet文件路径
            file_name = f"stock_{stock_code}_20240201.parquet"
            cache_path = os.path.join(cache_dir, file_name)
            
            # 读取现有parquet文件
            try:
                df = pd.read_parquet(cache_path, engine='fastparquet')
                if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                    df = df.set_index('date')
            except FileNotFoundError:
                print(f"[{idx}/{len(stock_codes)}] {stock_code}: parquet文件不存在，跳过")
                failed_count += 1
                continue
            
            # 准备新数据
            target_date_obj = pd.to_datetime(target_date, format='%Y%m%d')
            
            # 计算涨跌幅等指标
            if len(df) > 0:
                last_close = df['close'].iloc[-1]  # 前一交易日收盘价
                current_close = day_data['close']
                
                # 计算涨跌额和涨跌幅
                change_amount = current_close - last_close
                change_percent = (change_amount / last_close) * 100 if last_close != 0 else 0
                
                # 计算振幅
                amplitude = ((day_data['high'] - day_data['low']) / last_close) * 100 if last_close != 0 else 0
            else:
                change_amount = 0
                change_percent = 0
                amplitude = 0
            
            # 构建新数据行（按照原有格式）
            new_data = {
                "股票代码": stock_code[2:],  # 去掉sh/sz前缀
                "open": day_data['open'],
                "close": day_data['close'],
                "high": day_data['high'],
                "low": day_data['low'],
                "volume": day_data['volume'],
                "成交额": 0,  # miniqmt数据中没有成交额，设为0
                "振幅": round(amplitude, 2),
                "涨跌幅": round(change_percent, 2),
                "涨跌额": round(change_amount, 2),
                "换手率": 0  # miniqmt数据中没有换手率，设为0
            }
            
            # 检查是否已存在该日期的数据
            if target_date_obj in df.index:
                # 更新现有数据
                df.loc[target_date_obj, list(new_data.keys())] = list(new_data.values())
                print(f"[{idx}/{len(stock_codes)}] {stock_code}: 更新数据")
            else:
                # 添加新数据
                new_row = pd.DataFrame([new_data], index=[target_date_obj])
                df = pd.concat([df, new_row])
                print(f"[{idx}/{len(stock_codes)}] {stock_code}: 新增数据")
            
            # 保存更新后的数据
            df.to_parquet(cache_path, engine='fastparquet', compression='snappy')
            success_count += 1
            
        except Exception as e:
            print(f"[{idx}/{len(stock_codes)}] {stock_code}: 处理失败 - {e}")
            failed_count += 1
            continue
    
    print(f"\nparquet文件更新完成:")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")

def main():
    """主函数"""
    print("=" * 50)
    print("2025年10月24日股票数据下载和更新工具")
    print("=" * 50)
    
    # 步骤1: 获取所有股票代码
    print("\n步骤1: 获取股票代码列表...")
    stock_codes = get_all_stock_codes()
    if not stock_codes:
        print("无法获取股票代码，程序退出")
        return
    
    # 步骤2: 下载数据 ，数据已经下载过了
    # print(f"\n步骤2: 下载2025年10月24日数据...")
    # success_count, failed_stocks = download_stock_data_for_date(stock_codes, "20251024")
    
    # 步骤3: 更新parquet文件
    print(f"\n步骤3: 更新parquet缓存文件...")
    update_parquet_files_with_miniqmt_data("20251024")

    print("\n" + "=" * 50)
    print("数据下载和更新任务完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()