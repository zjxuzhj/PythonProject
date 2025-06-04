from datetime import datetime

from position_manage.portfolio import Portfolio
from position_manage.portfolio_db import save_portfolio, load_portfolio  # 修改导入
from position_manage.transaction import Transaction


def add_new_position(portfolio):
    try:
        transactions = [
            # Transaction(datetime(2025, 5, 28), "sz000710", "BUY", 14.15, 200),
            Transaction(datetime(2025, 5, 30), "sz002682", "SELL", 4.96, 200),
            # Transaction(datetime(2025, 5, 29), "sh600756", "BUY", 15.31, 200),
            # Transaction(datetime(2025, 5, 30), "sh600805", "BUY", 5.36, 1000),
            # Transaction(datetime(2025, 5, 30), "sh603680", "BUY", 10.64, 400),
            Transaction(datetime(2025, 6, 3), "sh603767", "SELL", 16.96, 200),
            # Transaction(datetime(2025, 6, 3), "sz002278", "BUY", 9.12, 400),
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


def generate_position_report(portfolio):
    """生成并打印持仓报告"""
    try:
        report = portfolio.get_position_report()
    except Exception as e:
        print(f"❌ 生成持仓报告失败: {str(e)}")
        return None


if __name__ == '__main__':
    try:
        portfolio = load_portfolio("portfolio.db")  # 加载数据库
        print("load data successfully")
    except:
        portfolio = Portfolio()
        print("create new db")

    add_new_position(portfolio)
    # generate_position_report(portfolio)
