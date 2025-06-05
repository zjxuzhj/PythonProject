from datetime import datetime

from position_manage.portfolio import Portfolio
from position_manage.portfolio_db import save_portfolio  # 修改导入
from position_manage.position_db_util import DBUtil
from position_manage.transaction import Transaction


def add_new_position(portfolio):
    try:
        transactions = [
            Transaction(datetime(2025, 6, 5), "sz002278", "SELL", 9.23, 400),
            Transaction(datetime(2025, 6, 5), "sh600805", "SELL", 5.21, 1000),
            Transaction(datetime(2025, 6, 5), "sz002682", "SELL", 5.05, 300),
            Transaction(datetime(2025, 6, 5), "sz000710", "SELL", 14.47, 100),

            Transaction(datetime(2025, 6, 5), "sh600743", "BUY", 2.2, 2200),
            Transaction(datetime(2025, 6, 5), "sh603127", "BUY", 19.25, 200),
            Transaction(datetime(2025, 6, 5), "sz002162", "BUY", 4.62, 1000),
        ]

        added_count = 0
        for t in transactions:
            if portfolio.add_transaction(t):  # 检查添加结果
                added_count += 1

        if added_count > 0:
            save_portfolio(portfolio)
            print(f"✅ 新增{added_count}笔持仓")
        else:
            print("⏩ 无新交易需要保存")

        return True
    except Exception as e:
        print(f"❌ 新增持仓失败: {str(e)}")
        return False

if __name__ == '__main__':
    db = DBUtil("portfolio.db")
    portfolio = Portfolio()
    portfolio.cash = db.get_cash()
    db.close()
    add_new_position(portfolio)
