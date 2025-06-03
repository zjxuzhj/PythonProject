import pandas as pd
import getAllStockCsv as stockCsv
from position_manage.position import Position


class Portfolio:
    def __init__(self):
        self.positions = {}
        self.cash = 1000000
        self.transaction_history = []

    def add_transaction(self, transaction):
        # 更新现金
        if transaction.action == 'BUY':
            self.cash -= transaction.price * transaction.shares
        else:
            self.cash += transaction.price * transaction.shares

        # 更新持仓
        query_tool = stockCsv.StockQuery()
        code = transaction.stock_code
        if code not in self.positions:
            self.positions[code] = Position(code, query_tool.get_name_by_code(code))

        if transaction.action == 'BUY':
            self.positions[code].buy_transactions.append(transaction)
        else:
            self.positions[code].sell_transactions.append(transaction)

        self.transaction_history.append(transaction)

    def get_position_report(self):
        """生成持仓报表（带异常处理）"""
        try:
            report = []
            for code, pos in self.positions.items():
                # 跳过已清仓股票
                if pos.total_shares <= 0:
                    continue

                report.append({
                    "股票代码": code,
                    "股票名称": pos.stock_name,
                    "持仓数量": pos.total_shares,
                    "平均成本": round(pos.avg_cost, 2),
                    "持有天数": pos.holding_period(),
                    "当前价值": 0  # 需通过行情API获取
                })
            return pd.DataFrame(report)

        except AttributeError as e:
            print(f"属性错误: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"生成报表失败: {e}")
            return pd.DataFrame()