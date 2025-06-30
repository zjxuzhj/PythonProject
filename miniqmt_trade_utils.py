# utils/trade_utils.py
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
import os
from xtquant import xtconstant

def can_cancel_order_status(status_code):
    # 定义一个包含所有可撤销状态的集合
    cancellable_statuses = {
        xtconstant.ORDER_UNREPORTED,
        xtconstant.ORDER_WAIT_REPORTING,
        xtconstant.ORDER_REPORTED,
        xtconstant.ORDER_PART_SUCC
    }
    # 判断传入的状态码是否在该集合中
    return status_code in cancellable_statuses


def save_trigger_prices_to_csv(trigger_prices):
    """将全局trigger_prices数据保存到CSV文件"""
    try:
        today_str = datetime.now().strftime('%Y-%m-%d')
        filename = os.path.join("output", f"trigger_prices_{today_str}.csv")

        # 转换数据结构为DataFrame
        all_data = []
        for stock_code, tiers in trigger_prices.items():
            for tier in tiers:
                all_data.append({
                    'date': today_str,
                    'stock_code': stock_code,
                    'price': tier['price'],
                    'weight': f"{tier['weight'] * 100}%",
                    'triggered': tier['triggered'],
                    'trigger_time': tier.get('trigger_time', '')  # 记录触发时间
                })

        if not all_data:
            print("⚠无触发价格数据需要保存")
            return

        df = pd.DataFrame(all_data)

        # 按股票代码和价格排序
        df = df.sort_values(['stock_code', 'price'], ascending=[True, False])

        # 保存到CSV（如果文件已存在则追加）
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            df = pd.concat([existing_df, df]).drop_duplicates(
                subset=['date', 'stock_code', 'price'],
                keep='last'
            )

        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"触发价格已保存: {filename}")

    except Exception as e:
        print(f"保存触发价格失败: {str(e)}")

def load_trigger_prices_from_csv(date_str=None):
    """从CSV加载指定日期的触发价格数据"""
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    try:
        filename = os.path.join("output", f"trigger_prices_{date_str}.csv")
        if not os.path.exists(filename):
            print(f"⚠未找到{date_str}的触发价格记录")
            return None

        df = pd.read_csv(filename)

        # 转换为全局trigger_prices格式
        loaded_data = defaultdict(list)
        for _, row in df.iterrows():
            loaded_data[row['stock_code']].append({
                'price': row['price'],
                'weight': row['weight'],
                'triggered': row['triggered'],
                'trigger_time': row.get('trigger_time', '')
            })

        return loaded_data

    except Exception as e:
        print(f"加载触发价格失败: {str(e)}")
        return None
