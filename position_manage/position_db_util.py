import sqlite3
from datetime import datetime


class DBUtil:
    def __init__(self, db_file="portfolio.db"):
        self.conn = sqlite3.connect(db_file)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        # 创建交易记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            stock_code TEXT NOT NULL,
            action TEXT NOT NULL,
            price REAL NOT NULL,
            shares INTEGER NOT NULL
        )
        ''')

        # 创建投资组合状态表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_state (
            id INTEGER PRIMARY KEY,
            cash REAL NOT NULL
        )
        ''')

        # 初始化现金状态
        cursor.execute('''
        INSERT OR IGNORE INTO portfolio_state (id, cash) 
        VALUES (1, 1000000)
        ''')
        self.conn.commit()

    def save_transaction(self, transaction):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO transactions (date, stock_code, action, price, shares)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            transaction.date.isoformat(),
            transaction.stock_code,
            transaction.action,
            transaction.price,
            transaction.shares
        ))
        self.conn.commit()

    def load_transactions(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT date, stock_code, action, price, shares FROM transactions')
        return cursor.fetchall()

    def close(self):
        self.conn.close()