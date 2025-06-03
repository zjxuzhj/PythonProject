class Transaction:
    def __init__(self, date, stock_code, action, price, shares):
        self.date = date  # 交易日期 (datetime对象)
        self.stock_code = stock_code  # 股票代码
        self.action = action  # 操作类型: 'BUY'/'SELL'
        self.price = price  # 交易价格
        self.shares = shares  # 交易数量