from position_manage.portfolio import Portfolio
from position_manage.portfolio_db import load_portfolio  # 修改导入

if __name__ == '__main__':
    try:
        portfolio = load_portfolio()  # 加载数据库
        print("load data successfully")
    except:
        portfolio = Portfolio()
        print("create new db")
    portfolio.get_position_report()
