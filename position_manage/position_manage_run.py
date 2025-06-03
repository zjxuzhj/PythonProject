from datetime import datetime
from position_manage.portfolio import Portfolio
from position_manage.portfolio_db import save_portfolio, load_portfolio  # 修改导入
from position_manage.transaction import Transaction

if __name__ == '__main__':
    # 尝试加载现有组合或创建新组合
    try:
        portfolio = load_portfolio("portfolio.db")  # 加载数据库
        print("成功从数据库加载现有投资组合")
    except:
        portfolio = Portfolio()
        print("创建新的投资组合")

    # 添加新交易
    portfolio.add_transaction(Transaction(
        date=datetime(2025, 6, 3),
        stock_code="sh601318",
        action="BUY",
        price=45.80,
        shares=500
    ))

    # 保存到数据库
    save_portfolio(portfolio, "portfolio.db")
    print("投资组合已保存到SQLite数据库")

    # 打印持仓报告
    report = portfolio.get_position_report()
    print("\n当前持仓报告:")
    print(report)