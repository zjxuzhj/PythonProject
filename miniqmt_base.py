import datetime
import os
import sys
import time

import pandas as pd
from xtquant import xtconstant
from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount

import getAllStockCsv as tools
query_tool = tools.StockQuery()

# 定义一个类 创建类的实例 作为状态的容器
class _a():
    pass


A = _a()
A.bought_list = []
A.hsa = xtdata.get_stock_list_in_sector('沪深A股')


def interact():
    """执行后进入repl模式"""
    import code
    code.InteractiveConsole(locals=globals()).interact()


# xtdata.download_sector_data()

class MyXtQuantTraderCallback(XtQuantTraderCallback):
    def on_disconnected(self):
        """
        连接断开
        :return:
        """
        print(datetime.datetime.now(), '连接断开回调')

    def on_stock_order(self, order):
        """
        委托回报推送
        :param order: XtOrder对象
        :return:
        """
        print(datetime.datetime.now(), '委托回调 投资备注', order.order_remark)

    def on_stock_trade(self, trade):
        """
        成交变动推送
        :param trade: XtTrade对象
        :return:
        """
        print(datetime.datetime.now(), '成交回调', trade.order_remark,
              f"委托方向(48买 49卖) {trade.offset_flag} 成交价格 {trade.traded_price} 成交数量 {trade.traded_volume}")

    def on_order_error(self, order_error):
        """
        委托失败推送
        :param order_error:XtOrderError 对象
        :return:
        """
        # print("on order_error callback")
        # print(order_error.order_id, order_error.error_id, order_error.error_msg)
        print(f"委托报错回调 {order_error.order_remark} {order_error.error_msg}")

    def on_cancel_error(self, cancel_error):
        """
        撤单失败推送
        :param cancel_error: XtCancelError 对象
        :return:
        """
        print(datetime.datetime.now(), sys._getframe().f_code.co_name)

    def on_order_stock_async_response(self, response):
        """
        异步下单回报推送
        :param response: XtOrderResponse 对象
        :return:
        """
        print(f"异步委托回调 投资备注: {response.order_remark}")

    def on_cancel_order_stock_async_response(self, response):
        """
        :param response: XtCancelOrderResponse 对象
        :return:
        """
        print(datetime.datetime.now(), sys._getframe().f_code.co_name)

    def on_account_status(self, status):
        """
        :param response: XtAccountStatus 对象
        :return:
        """
        print(datetime.datetime.now(), sys._getframe().f_code.co_name)


def order_stock(stock, target_amount=5000):
    full_tick = xtdata.get_full_tick([stock])
    print(f"{stock} 全推行情： {full_tick}")
    current_price = full_tick[stock]['lastPrice']
    # 买入金额 取目标金额 与 可用金额中较小的
    buy_amount = min(target_amount, available_cash)
    # 买入数量 取整为100的整数倍
    buy_vol = int(buy_amount / current_price / 100) * 100
    print(f"当前可用资金 {available_cash} 目标买入金额 {target_amount} 买入股数 {buy_vol}股")
    async_seq = xt_trader.order_stock_async(acc, stock, xtconstant.STOCK_BUY, buy_vol, xtconstant.FIX_PRICE,
                                            current_price,
                                            'strategy_name', stock)


def pre_order_stock(stock, target_amount=5000, pre_price=0.0):
    # 买入金额 取目标金额 与 可用金额中较小的
    buy_amount = min(target_amount, available_cash)
    # 买入数量 取整为100的整数倍
    buy_vol = int(buy_amount / pre_price / 100) * 100
    print(f"当前可用资金 {available_cash} 目标买入金额 {target_amount} 买入股数 {buy_vol}股")
    async_seq = xt_trader.order_stock_async(acc, stock, xtconstant.STOCK_BUY, buy_vol, xtconstant.FIX_PRICE,
                                            pre_price,
                                            'strategy_name', stock)


def can_cancel_order_status(status_code):
    # 定义一个包含所有可撤销状态的集合
    cancellable_statuses = {
        xtconstant.ORDER_UNREPORTED,
        xtconstant.ORDER_WAIT_REPORTING,
        xtconstant.ORDER_REPORTED,
        xtconstant.ORDER_PART_SUCC
    }
    # 判断传入的状态码是否在该集合中
    return status_code in cancellable_statuses


def query_and_cancel_specific_orders(stock_code):
    """
    查询指定股票的所有挂单并撤单
    :param stock_code: 股票代码（如'600000.SH'）
    """
    # 查询当日所有委托单
    orders = xt_trader.query_stock_orders(acc)
    target_orders = []

    # 筛选指定股票的未成交挂单（状态码0表示可撤单状态）
    for order in orders:
        if (order.stock_code == stock_code
                and can_cancel_order_status(order.order_status)
                and order.order_volume > order.traded_volume):  # 存在未成交部分
            target_orders.append(order)

    if not target_orders:
        print(f"{stock_code} 无未成交挂单")
        return

    # 批量撤单
    success_count = 0
    for order in target_orders:
        # 打印挂单详情
        print(f"发现挂单：{order.stock_code} {order.order_type} "
              f"数量:{order.order_volume - order.traded_volume}股 @ {order.price}")

        # 执行异步撤单[9](@ref)
        cancel_result = xt_trader.cancel_order_stock_async(
            acc,
            order.order_id  # 使用原始订单ID撤单
        )

        # 处理撤单结果（0表示成功）
        if cancel_result != -1:
            success_count += 1
            print(f"撤单成功 | ID:{order.order_id}")
        else:
            print(f"撤单失败 | ID:{order.order_id} 错误码:{cancel_result}")

    print(f"操作完成：成功撤单 {success_count}/{len(target_orders)} 笔")


def get_stock_data(symbol):
    """带本地缓存的数据获取"""
    file_name = f"stock_{symbol}_20240201.parquet"
    cache_path = os.path.join("data_cache", file_name)

    # 非强制更新时尝试读取缓存
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path, engine='fastparquet')
            print(f"从缓存加载数据：{symbol}")
            return df, True
        except Exception as e:
            print(f"缓存读取失败：{e}（建议删除损坏文件：{cache_path}）")

    # 强制更新或缓存不存在时获取新数据（网页7）
    print(f"数据获取失败：{symbol}")
    return pd.DataFrame()


def get_guess_ma5_price(stock_code):
    """获取指定股票的最新MA5价格"""
    pure_code = tools.convert_stock_code(stock_code)  # 如603722.SH -> sh603722
    df, _ = get_stock_data(pure_code)

    if df.empty or len(df) < 5:
        print(f"警告：{stock_code} 数据不足，无法计算MA5")
        return None

    try:
        modified_df = modify_last_days_and_calc_ma5(df)
        return modified_df['MA5'].iloc[-1]  # 返回最新MA5值
    except ValueError as e:
        print(f"MA5计算失败：{e}")
        return None

def get_ma5_price(stock_code):
    """获取指定股票的最新MA5价格"""
    pure_code = tools.convert_stock_code(stock_code)  # 如603722.SH -> sh603722
    df, _ = get_stock_data(pure_code)

    if df.empty or len(df) < 5:
        print(f"警告：{stock_code} 数据不足，无法计算MA5")
        return None

    try:
        # 核心计算逻辑（网页1、网页3、网页6）
        df_sorted = df.sort_index(ascending=True)  # 确保时间升序排列
        df_sorted['MA5'] = df_sorted['close'].rolling(
            window=5,
            min_periods=5  # 严格要求5个有效数据点
        ).mean().round(2)  # 保留两位小数

        return df_sorted['MA5'].iloc[-1]  # 返回最新MA5值

    except KeyError as e:
        print(f"数据列缺失错误：{str(e)}")
        return None
    except Exception as e:
        print(f"MA5计算异常：{str(e)}")
        return None

def modify_last_days_and_calc_ma5(df):
    if df.empty or len(df) < (2):
        raise ValueError(f"数据不足，至少需要{2}个交易日数据")

    modified_df = df.copy().sort_index(ascending=True)
    modified_df['adjusted_close'] = modified_df['close']

    # 复制最后一行并调整收盘价
    new_row = modified_df.iloc[-1].copy()
    new_row['close'] *= 1.06
    new_row.name = new_row.name + pd.Timedelta(days=1)  # 日期顺延一日
    modified_df = pd.concat([modified_df, new_row.to_frame().T], axis=0)

    # 计算MA5
    modified_df['MA5'] = modified_df['close'].rolling(
        window=5,
        min_periods=1
    ).mean().round(2)
    return modified_df


def auto_order_by_ma5(stock_code, target_amount=5000):
    ma5_price = get_guess_ma5_price(stock_code)
    if ma5_price is None:
        return False

    pre_order_stock(
        stock=stock_code,
        target_amount=target_amount,
        pre_price=ma5_price  # 传入MA5价格
    )
    return True


def check_ma5_breach():
    """检测持仓中跌破五日线的股票"""
    breach_list = []

    # 获取可用持仓字典
    positions = xt_trader.query_stock_positions(acc)
    position_available_dict = {pos.stock_code: pos.m_nCanUseVolume for pos in positions}

    for stock_code, hold_vol in position_available_dict.items():
        if hold_vol <= 0:
            continue

        try:
            # 获取当前价格
            tick = xtdata.get_full_tick([stock_code])[stock_code]
            current_price = tick['lastPrice']

            # 获取MA5价格
            ma5_price = get_ma5_price(stock_code)
            if ma5_price is None:
                continue

            # 价格对比（保留两位小数）
            if round(current_price, 2) < round(ma5_price, 2):
                # 获取股票名称
                stock_name = query_tool.get_name_by_code(stock_code)

                breach_list.append({
                    '代码': stock_code,
                    '名称': stock_name,
                    '持有数量': hold_vol,
                    '当前价格': current_price,
                    'MA5价格': ma5_price
                })
        except Exception as e:
            print(f"检测异常 {stock_code}: {str(e)}")
            continue

    return breach_list

if __name__ == "__main__":
    xtdata.enable_hello = False
    path = r'D:\备份\国金证券QMT交易端\userdata_mini'
    session_id = int(time.time())
    xt_trader = XtQuantTrader(path, session_id)

    # 创建资金账号为 800068 的证券账号对象 股票账号为STOCK 信用CREDIT 期货FUTURE
    acc = StockAccount('8886969255', 'STOCK')
    # 创建交易回调类对象，并声明接收回调
    callback = MyXtQuantTraderCallback()
    xt_trader.register_callback(callback)
    # 启动交易线程
    xt_trader.start()
    # 建立交易连接，返回0表示连接成功
    connect_result = xt_trader.connect()
    print('建立交易连接，返回0表示连接成功', connect_result)
    # 对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功
    subscribe_result = xt_trader.subscribe(acc)
    print('对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功', subscribe_result)
    # 取账号信息
    account_info = xt_trader.query_stock_asset(acc)
    # 取可用资金
    available_cash = account_info.m_dCash

    print(acc.account_id, '可用资金', available_cash)
    # 查账号持仓
    positions = xt_trader.query_stock_positions(acc)
    # 取各品种 总持仓 可用持仓
    position_total_dict = {i.stock_code: i.m_nVolume for i in positions}
    position_available_dict = {i.stock_code: i.m_nCanUseVolume for i in positions}
    print(acc.account_id, '持仓字典', position_total_dict)
    print(acc.account_id, '可用持仓字典', position_available_dict)

    breach_stocks = check_ma5_breach()

    if breach_stocks:
        print("\n跌破五日线持仓预警（截至%s）" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        df = pd.DataFrame(breach_stocks)
        print(df[['代码', '名称', '持有数量', '当前价格']].to_string(index=False))
    else:
        print("\n当前无持仓跌破五日线")

    # target_stocks = ['603722.SH']
    # for stock_code in target_stocks:
    #     success = auto_order_by_ma5(stock_code, 5000)
    #     if not success:
    #         print(f"【风控拦截】{stock_code} 下单失败，请检查数据完整性")

    # pre_order_stock( '603722.SH',5000,42.15)
    # pre_order_stock( '603725.SH',10000,8.19)
    # pre_order_stock( '002923.SZ',10000,14.22)
    # query_and_cancel_specific_orders('603722.SH')
    # query_and_cancel_specific_orders('002923.SZ')
    # 查询当日所有的委托
    # print("query orders:")
    # orders = xt_trader.query_stock_orders(acc)
    # print("orders:", len(orders))
    # if len(orders) != 0:
    #     print("last order:")
    #     print("{0} {1} {2}".format(orders[-1].stock_code, orders[-1].order_volume, orders[-1].price))
    #     order_id_to_cancel = orders[-1].order_id  # 要撤销的订单号
    #     cancel_result = xt_trader.cancel_order_stock_async(acc, order_id_to_cancel)
    #     print(f"撤单结果: {cancel_result}")
