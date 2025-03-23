import akshare as ak
import sqlite3
import pandas as pd
from datetime import datetime
# import schedule
import time

# 数据库配置
DB_PATH = "stock_data.db"
TABLE_NAME = "stock_spot"


def init_database():
    """初始化数据库表结构"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        date DATE NOT NULL,
        code TEXT NOT NULL,
        name TEXT,
        latest_price REAL,
        turnover REAL,
        change_percent REAL,
        PRIMARY KEY (date, code)
    )
    """
    cursor.execute(create_table_sql)
    conn.commit()
    conn.close()


def save_to_database(df):
    """数据入库（含去重逻辑）"""
    conn = sqlite3.connect(DB_PATH)
    df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')

    # 字段名统一为英文
    df = df.rename(columns={'日期': 'date'})

    # 使用批量插入
    df.to_sql(
        name=TABLE_NAME,
        con=conn,
        if_exists='append',
        index=False,
        dtype={
            'date': 'DATE',
            'code': 'TEXT',
            'name': 'TEXT',
            'latest_price': 'REAL',
            'turnover': 'REAL',
            'change_percent': 'REAL'
        },
        chunksize=1000,
        method='multi'
    )

    # 手动去重（基于主键）
    cursor = conn.cursor()
    cursor.execute(f"""
        DELETE FROM {TABLE_NAME}
        WHERE rowid NOT IN (
            SELECT MIN(rowid)
            FROM {TABLE_NAME}
            GROUP BY date, code
        )
    """)
    conn.commit()
    conn.close()


def fetch_and_save():
    """获取数据并存储"""
    try:
        df = ak.stock_zh_a_spot()
        df['日期'] = datetime.now().strftime('%Y-%m-%d')

        df = df.rename(columns={
            '代码': 'code',
            '名称': 'name',
            '最新价': 'latest_price',
            '成交额': 'turnover',
            '涨跌幅': 'change_percent'
        })[['日期', 'code', 'name', 'latest_price', 'turnover', 'change_percent']]

        save_to_database(df)
        print(f"{datetime.now()} 数据更新成功，新增记录：{len(df)}条")
    except Exception as e:
        print(f"数据获取失败：{str(e)}")


if __name__ == "__main__":
    init_database()
    fetch_and_save()  # 首次执行

    # 设置每天15:30定时执行
    # schedule.every().day.at("15:30").do(fetch_and_save)

    try:
        while True:
            # schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序已安全退出")