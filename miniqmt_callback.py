import sys
from datetime import datetime
from xtquant import xtconstant
from xtquant.xttrader import XtQuantTraderCallback

from position_manage.portfolio import Portfolio
from position_manage.transaction import Transaction
from position_manage.portfolio_db import save_portfolio
import getAllStockCsv as tools


def convert_unix_timestamp(timestamp):
    """
    将Unix时间戳转换为datetime对象（支持秒级/毫秒级时间戳）
    参数:
        timestamp: int - Unix时间戳（秒级或毫秒级）
    返回:
        datetime - 本地时区的datetime对象
    """
    if len(str(timestamp)) > 10:
        return datetime.fromtimestamp(timestamp / 1000.0)
    else:
        return datetime.fromtimestamp(timestamp)


# 添加数据库保存函数
def save_transaction_to_db(trade, trade_type):
    try:
        transaction = Transaction(
            date=convert_unix_timestamp(trade.traded_time),
            stock_code=tools.convert_stock_code(trade.stock_code),
            action=trade_type,
            price=trade.traded_price,
            shares=trade.traded_volume
        )
        portfolio = Portfolio()
        portfolio.add_transaction(transaction)
        save_portfolio(portfolio)
        print(f"交易记录已保存: {trade_type} {trade.stock_code} {trade.traded_volume}股 @ {trade.traded_price}")
    except Exception as e:
        print(f"保存交易记录失败: {str(e)}")


class MyXtQuantTraderCallback(XtQuantTraderCallback):
    def __init__(self, query_tool, trigger_prices_ref, save_func_ref):
        """
        初始化回调类
        :param query_tool: 股票查询工具
        """
        self.query_tool = query_tool
        self.trigger_prices = trigger_prices_ref  # 保存对全局 trigger_prices 的引用
        self.save_trigger_prices_to_csv = save_func_ref  # 保存对保存函数的引用

    def on_disconnected(self):
        print(datetime.now(), '连接断开回调')

    def on_stock_order(self, order):
        order_str = "撤单成功" if order.order_status == xtconstant.ORDER_CANCELED else "挂单成功"
        stock_name = self.query_tool.get_name_by_code(order.stock_code)
        print(f"{order_str}！名称：{stock_name}, 代码: {order.stock_code}, "
              f"委托类型: {order.order_type}, 委托数量: {order.order_volume}, 委托价格: {order.price:.2f}")

    def on_stock_trade(self, trade):
        trade_type = "BUY" if trade.offset_flag == xtconstant.OFFSET_FLAG_OPEN else "SELL"
        print(datetime.now(), '成交回调', trade.order_remark,
              f", 委托方向: {'买入' if trade_type == 'BUY' else '卖出'}, "
              f"成交价格 {trade.traded_price} 成交数量 {trade.traded_volume}")

        if trade_type == "BUY":
            # 检查这个成交是否对应我们预设的某个触发档位
            stock_code = trade.stock_code
            traded_price = trade.traded_price

            if stock_code in self.trigger_prices:
                matched_and_updated = False
                for tier in self.trigger_prices[stock_code]:
                    if not tier.get('triggered', False) and abs(tier['price'] - traded_price) < 0.01:
                        print(f"匹配到预设档位！正在更新状态: {stock_code} @ {tier['price']:.2f}")
                        tier['triggered'] = True
                        tier['trigger_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        matched_and_updated = True
                        break

                if matched_and_updated:
                    # 使用传入的函数引用来保存
                    self.save_trigger_prices_to_csv(self.trigger_prices)
                    print(f"√ {stock_code} 的触发状态已更新并保存。")

        # 保存交易记录到数据库
        save_transaction_to_db(trade, trade_type)

    def on_order_error(self, order_error):
        print(f"委托报错回调 {order_error.order_remark} {order_error.error_msg}")

    def on_cancel_error(self, cancel_error):
        print(datetime.now(), sys._getframe().f_code.co_name)

    def on_order_stock_async_response(self, response):
        print(f"异步委托回调 投资备注: {response.order_remark}")

    def on_cancel_order_stock_async_response(self, response):
        print(datetime.now(), sys._getframe().f_code.co_name)

    def on_account_status(self, status):
        print(datetime.now(), sys._getframe().f_code.co_name)
