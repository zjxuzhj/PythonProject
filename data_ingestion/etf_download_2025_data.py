"""
ETF动量轮动策略 - 2025年历史数据下载脚本
下载etf_momentum_rotation_qmt.py中所需的所有ETF的2025年日线数据
"""

import time
import xtquant.xtdata as xtdata
import pandas as pd
from datetime import datetime

# ETF池配置 - 从etf_momentum_rotation_qmt.py中提取
ETF_POOL = [
    # 境外
    "513100.SH",  # 纳指ETF 2013.4 168亿
    "159509.SZ",  # 纳指科技ETF 2023.7 121亿
    "513520.SH",  # 日经ETF 2019.6 17.6亿
    "513030.SH",  # 德国ETF 2014.8 17.3亿
    # 国内
    "513130.SH",  # 恒生科技 2021.5 407亿
    "510180.SH",  # 上证180 2006.4 236亿
    "159915.SZ",  # 创业板ETF 2011.9 1011亿
    # "513160.SH",  # 恒科30
    # "513690.SH",  # 港股红利
    # "510880.SH",  # 红利ETF
    # "510300.SH",  # 300ETF

    # 行业ETF
    "159851.SZ",  # 金融科技 2021.3 123亿
    "588120.SH",  # 科创100 2023.9 11.5亿
    "512290.SH",  # 生物医药 2019.4 32亿
    "515070.SH",  # 人工智能ETF 2019.12 92亿
    "159755.SZ",  # 电池ETF 2021.6 170亿
    "512480.SH",  # 半导体 2019.6 230亿
    "516160.SH",  # 新能源 2021.2 61亿 中证新能

    # "510410.SH",  # 资源 2012.4 6亿
    # "515650.SH",  # 消费50 2019.10 31亿
    # "159796.SZ",  # 电池50ETF 2022.3 99.3亿
    # "515030.SH",  # 新能源车 2020.2 59亿 中证新能车
    # "513050.SH",  # 中概互联网ETF 2017.1 393亿
    # "517200.SH", # 互联网ETF
    # "512710.SH",  # 军工ETF 2019.7 141亿
    # "159692.SZ",  # 证券 2023.5 9亿
    # "515880.SH",  # 通信ETF

    # 防守
    # 商品
    "518880.SH",  # 黄金ETF 2013.7 714亿
    "159985.SZ",  # 豆粕ETF 2019.9 24亿
    "501018.SH",  # 南方原油 2016.6 4.6亿
    # "159980.SZ",  # 有色ETF 2019.10 21.8亿

    # 债券
    "511090.SH",  # 30年国债ETF 2023.5 310亿
    # "159001.SZ",  # 货币ETF
]

# 时间范围设置 - 2025年全年
start_date = "20260101"
end_date = "20260119"
period = "1d"  # 日线数据

def download_etf_data():
    """下载所有ETF的2025年日线数据"""
    total_etfs = len(ETF_POOL)
    print(f"准备下载 {total_etfs} 只ETF的 '{period}' 周期数据...")
    print(f"时间范围: {start_date} -> {end_date}")
    print("=" * 50)
    
    success_count = 0
    failed_etfs = []
    
    for idx, etf_code in enumerate(ETF_POOL, 1):
        print(f"[{idx}/{total_etfs}] 开始下载: {etf_code}")
        
        try:
            # 使用增量下载模式，自动检查本地数据并只下载需要补充的部分
            xtdata.download_history_data2(
                stock_list=[etf_code],
                period=period,
                start_time=start_date,
                end_time=end_date,
                incrementally=True,  # 增量下载，支持断点续传
                callback=lambda data: print(f"  -> 进度: {data['finished']}/{data['total']} {data['stockcode']}")
            )
            print(f"[{idx}/{total_etfs}] ✅ 完成下载: {etf_code}")
            success_count += 1
            
        except Exception as e:
            print(f"[{idx}/{total_etfs}] ❌ 下载失败: {etf_code}, 错误: {e}")
            failed_etfs.append((etf_code, str(e)))
        
        # 避免请求过于频繁
        time.sleep(0.5)
    
    print("=" * 50)
    print(f"下载完成！成功: {success_count}/{total_etfs}")
    
    if failed_etfs:
        print(f"\n失败的ETF ({len(failed_etfs)}个):")
        for etf_code, error in failed_etfs:
            print(f"  - {etf_code}: {error}")
    
    return success_count, failed_etfs

def verify_downloaded_data():
    """验证下载的数据完整性"""
    print("\n" + "=" * 50)
    print("开始验证下载的数据...")
    
    verification_results = []
    
    for idx, etf_code in enumerate(ETF_POOL, 1):
        print(f"[{idx}/{len(ETF_POOL)}] 验证: {etf_code}")
        
        try:
            # 获取本地数据
            data = xtdata.get_local_data(
                field_list=['open', 'high', 'low', 'close', 'volume'],
                stock_list=[etf_code],
                period='1d',
                start_time=start_date,
                end_time=end_date,
            )
            
            if etf_code in data and not data[etf_code].empty:
                df = data[etf_code]
                data_count = len(df)
                
                # 获取数据的时间范围
                if not df.empty:
                    # 将索引转换为字符串然后转换为日期
                    df_copy = df.copy()
                    df_copy.index = pd.to_datetime(df_copy.index.astype(str), format='%Y%m%d')
                    start_actual = df_copy.index.min().strftime('%Y%m%d')
                    end_actual = df_copy.index.max().strftime('%Y%m%d')
                    
                    print(f"  ✅ 数据完整: {data_count} 条记录, 时间范围: {start_actual} ~ {end_actual}")
                    verification_results.append({
                        'etf_code': etf_code,
                        'status': 'success',
                        'count': data_count,
                        'start_date': start_actual,
                        'end_date': end_actual
                    })
                else:
                    print(f"  ❌ 数据为空")
                    verification_results.append({
                        'etf_code': etf_code,
                        'status': 'empty',
                        'count': 0
                    })
            else:
                print(f"  ❌ 未找到本地数据")
                verification_results.append({
                    'etf_code': etf_code,
                    'status': 'not_found',
                    'count': 0
                })
                
        except Exception as e:
            print(f"  ❌ 验证失败: {e}")
            verification_results.append({
                'etf_code': etf_code,
                'status': 'error',
                'error': str(e)
            })
    
    # 汇总验证结果
    print("\n" + "=" * 50)
    print("验证结果汇总:")
    
    success_count = sum(1 for r in verification_results if r['status'] == 'success')
    print(f"成功验证: {success_count}/{len(ETF_POOL)}")
    
    if success_count < len(ETF_POOL):
        print("\n问题ETF:")
        for result in verification_results:
            if result['status'] != 'success':
                print(f"  - {result['etf_code']}: {result['status']}")
    
    return verification_results

if __name__ == "__main__":
    print("ETF动量轮动策略 - 2025年历史数据下载")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 下载数据
    success_count, failed_etfs = download_etf_data()
    
    # 2. 验证数据
    verification_results = verify_downloaded_data()
    
    print("\n" + "=" * 50)
    print("任务完成！")
    print(f"下载成功: {success_count}/{len(ETF_POOL)}")
    
    success_verify = sum(1 for r in verification_results if r['status'] == 'success')
    print(f"验证成功: {success_verify}/{len(ETF_POOL)}")