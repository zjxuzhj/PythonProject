from position_manage.position_db_util import DBUtil
from position_manage.portfolio import Portfolio
from position_manage.transaction import Transaction
from datetime import datetime


def save_portfolio(portfolio, db_file="portfolio.db"):
    """保存投资组合到SQLite数据库"""
    db = DBUtil(db_file)
    try:
        # 保存现金状态
        db.update_cash(portfolio.cash)

        # 保存所有交易记录
        for t in portfolio.transaction_history:
            db.save_transaction(t)
    finally:
        db.close()


def load_portfolio(db_file="portfolio.db"):
    """从SQLite数据库加载投资组合"""
    db = DBUtil(db_file)
    portfolio = Portfolio()
    try:
        # 加载现金
        portfolio.cash = db.get_cash()

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