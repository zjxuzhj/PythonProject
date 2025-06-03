from datetime import datetime


class Position:
    def __init__(self, stock_code, stock_name):
        self.stock_code = stock_code
        self.stock_name = stock_name
        self.buy_transactions = []  # 买入交易记录
        self.sell_transactions = []  # 卖出交易记录

    @property
    def total_shares(self):
        """计算当前持仓数量"""
        bought = sum(t.shares for t in self.buy_transactions)
        sold = sum(t.shares for t in self.sell_transactions)
        return bought - sold

    @property
    def avg_cost(self):
        """计算平均成本价"""
        total_cost = sum(t.price * t.shares for t in self.buy_transactions)
        total_shares = sum(t.shares for t in self.buy_transactions)
        return total_cost / total_shares if total_shares > 0 else 0

    def holding_period(self):
        """计算持有周期（天）"""
        if not self.buy_transactions: return 0
        first_buy = min(t.date for t in self.buy_transactions)
        return (datetime.now() - first_buy).days