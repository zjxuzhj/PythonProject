from datetime import datetime

from position_manage.portfolio import Portfolio
from position_manage.portfolio_db import save_portfolio  # 修改导入
from position_manage.position_db_util import DBUtil
from position_manage.transaction import Transaction


def add_new_position(portfolio):
    try:
        month = 6
        day = 17
        transactions = [
            Transaction(datetime(2025, month, day), "sh603725", "SELL", 7.98, 600),
            Transaction(datetime(2025, month, day), "sz002923", "SELL", 13.69, 500),
            Transaction(datetime(2025, month, day), "sh605018", "SELL", 12.1, 300),
            # Transaction(datetime(2025, month, day), "sz000530", "SELL", 5.94, 800),
            # Transaction(datetime(2025, month, day), "sh600830", "SELL", 9.8, 500),

            Transaction(datetime(2025, month, day), "sz002449", "BUY", 9.94, 1000),
            Transaction(datetime(2025, month, day), "sz002246", "BUY", 13.9, 700),
            Transaction(datetime(2025, month, day), "sh603353", "BUY", 17.64, 500),
            # Transaction(datetime(2025, month, day), "sh605066", "BUY", 7.96, 600),
            # Transaction(datetime(2025, month, day), "sz002335", "BUY", 43.21, 100),
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
