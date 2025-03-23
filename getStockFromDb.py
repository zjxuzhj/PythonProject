import akshare as ak
import sqlite3
import pandas as pd
from datetime import datetime
# import schedule
import time

#### 从数据库获得当天股票数据

# 数据库配置
DB_PATH = "stock_data.db"
TABLE_NAME = "stock_spot"

def export_top20_stocks():
    """导出成交额TOP20股票到Excel"""
    try:
        # 连接数据库
        conn = sqlite3.connect(DB_PATH)

        # 获取最新交易日期
        max_date = pd.read_sql("SELECT MAX(date) FROM stock_spot", conn).iloc[0, 0]

        # 查询当日成交额TOP20
        query = f"""
        SELECT 
            code AS 股票代码,
            name AS 股票名称,
            latest_price AS 最新价格,
            turnover/1e8 AS 成交金额_亿元,
            ROUND(change_percent, 2) AS 涨跌幅_百分比
        FROM {TABLE_NAME}
        WHERE date = '{max_date}'
        ORDER BY turnover DESC
        LIMIT 20
        """

        top20_df = pd.read_sql(query, conn)

        # 导出Excel
        filename = f"成交额TOP20_{max_date}.xlsx"
        top20_df.to_excel(filename, index=False, engine='openpyxl')

        print(f"成功导出：{filename}")
        print("数据预览：")
        print(top20_df.head())

    except Exception as e:
        print(f"导出失败：{str(e)}")
    finally:
        conn.close()


if __name__ == "__main__":
    # 添加导出功能
    export_top20_stocks()  # 新增的导出函数

    # ... 其他定时任务代码 ...