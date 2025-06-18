import os
import threading
from datetime import datetime

from position_manage.portfolio import Portfolio
from position_manage.position_db_util import DBUtil
from position_manage.transaction import Transaction

db_lock = threading.Lock()


def save_portfolio(portfolio):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "position_manage")
    os.makedirs(target_dir, exist_ok=True)
    db_path = os.path.join(target_dir, "portfolio.db")
    with db_lock:
        try:
            db = DBUtil(db_path)  # 传递绝对路径
            db.conn.execute("BEGIN TRANSACTION")
            for t in portfolio.transaction_history:
                db.save_transaction(t)
            db.conn.commit()
            return True
        except Exception as e:
            db.conn.rollback()
            print(f"❌ 保存失败: {str(e)}")
            return False
        finally:
            db.close()


def load_portfolio():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "position_manage")
    os.makedirs(target_dir, exist_ok=True)
    db_path = os.path.join(target_dir, "portfolio.db")
    db = DBUtil(db_path)
    portfolio = Portfolio()
    try:
        # 加载交易记录
        transactions = db.load_transactions()
        for t_data in transactions:
            transaction = Transaction(
                date=datetime.fromisoformat(t_data[0]),
                stock_code=t_data[1],
                action=t_data[2],
                price=t_data[3],
                shares=t_data[4]
            )
            portfolio.load_transaction(transaction)
    finally:
        db.close()
    return portfolio
