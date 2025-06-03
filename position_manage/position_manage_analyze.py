from position_manage.portfolio import Portfolio
from position_manage.portfolio_db import load_portfolio  # 修改导入


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

    generate_position_report(portfolio)
