from datetime import datetime

from position_manage.portfolio import Portfolio
from position_manage.portfolio_db import save_portfolio, load_portfolio  # 修改导入
from position_manage.transaction import Transaction


def add_new_position(portfolio):
    try:
        transactions = [
            Transaction(datetime(2025, 6, 5), "sz000001", "BUY", 20.2, 1000),
            Transaction(datetime(2025, 6, 5), "sh600036", "BUY", 20.50, 800)
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
        if not report.empty:
            print("\n📊 当前持仓报告:")
            print(report)
            return report
        else:
            print("ℹ️ 当前无持仓记录")
            return None
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
