from datetime import datetime

from position_manage.portfolio import Portfolio
from position_manage.portfolio_db import save_portfolio  # 修改导入
from position_manage.position_db_util import DBUtil
from position_manage.transaction import Transaction


def add_new_position(portfolio):
    try:
        month = 6
        day = 9
        transactions = [
            Transaction(datetime(2025, month, day), "sh600743", "SELL", 2.18, 1100),
            Transaction(datetime(2025, month, day), "sz002657", "SELL", 24.5, 200),
            Transaction(datetime(2025, month, day), "sz000555", "SELL", 12.41, 500),
            # Transaction(datetime(2025, month, day), "sz000710", "SELL", 14.47, 100),

            Transaction(datetime(2025, month, day), "sh603305", "BUY", 13.73, 300),
            # Transaction(datetime(2025, month, day), "sz002612", "BUY", 18.57, 500),
            # Transaction(datetime(2025, month, day), "sz002657", "BUY", 25.52, 200),
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
