import logging
import os
import sys
import threading
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from xtquant import xtconstant
from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount

import getAllStockCsv as tools
from position_manage.portfolio import Portfolio
from position_manage.portfolio_db import save_portfolio
from position_manage.transaction import Transaction

query_tool = tools.StockQuery()


# 定义一个类 创建类的实例 作为状态的容器
class _a():
    pass


A = _a()
A.bought_list = []
A.hsa = xtdata.get_stock_list_in_sector('沪深A股')


def setup_logger():
    """配置日志记录器"""
    logger = logging.getLogger("QMT_Strategy")
    logger.setLevel(logging.INFO)

    # 创建按天轮转的日志处理器 [4,8](@ref)
    log_handler = TimedRotatingFileHandler(
        "qmt_strategy.log",
        when="midnight",
        interval=1,
        backupCount=7
    )

    # 设置日志格式 [1,5](@ref)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    log_handler.setFormatter(formatter)

    # 同时输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(log_handler)
    logger.addHandler(console_handler)
    return logger


# 创建状态监控函数
def monitor_strategy_status(logger):
    """每30分钟记录程序状态"""
    while True:
        try:
            # 1. 记录基本状态信息
            status_msg = (
                f"策略运行正常 | "
                f"可用资金: {available_cash:.2f} | "
                f"持仓数量: {len(hold_stocks)}"
            )
            logger.info(status_msg)

            # 4. 记录挂单情况（可选）
            orders = xt_trader.query_stock_orders(acc)
            active_orders = [o for o in orders if can_cancel_order_status(o.order_status)]
            logger.info(f"活跃挂单数量: {len(active_orders)}")

        except Exception as e:
            logger.error(f"状态监控异常: {str(e)}")

        # 等待30分钟 [7](@ref)
        time.sleep(30 * 60)


def interact():
    """执行后进入repl模式"""
    import code
    code.InteractiveConsole(locals=globals()).interact()


def convert_unix_timestamp(timestamp):
    """
    将Unix时间戳转换为datetime对象（支持秒级/毫秒级时间戳）

    参数:
        timestamp: int - Unix时间戳（秒级或毫秒级）

    返回:
        datetime - 本地时区的datetime对象
    """
    # 判断时间戳长度（毫秒级时间戳通常为13位）
    if len(str(timestamp)) > 10:
        # 毫秒级时间戳：截取前10位转换为秒
        return datetime.fromtimestamp(timestamp / 1000.0)
    else:
        # 秒级时间戳直接转换
        return datetime.fromtimestamp(timestamp)


# 添加数据库保存函数
def save_transaction_to_db(trade, trade_type):
    """将交易记录保存到数据库"""
    try:
        # 创建交易记录对象
        transaction = Transaction(
            date=convert_unix_timestamp(trade.traded_time),
            stock_code=tools.convert_stock_code(trade.stock_code),
            action=trade_type,
            price=trade.traded_price,
            shares=trade.traded_volume
        )

        # 创建投资组合对象并添加交易
        portfolio = Portfolio()
        portfolio.add_transaction(transaction)

        # 保存到数据库
        save_portfolio(portfolio)
        print(f"✅ 交易记录已保存: {trade_type} {trade.stock_code} {trade.traded_volume}股 @ {trade.traded_price}")
    except Exception as e:
        print(f"❌ 保存交易记录失败: {str(e)}")


# xtdata.download_sector_data()

class MyXtQuantTraderCallback(XtQuantTraderCallback):
    def on_disconnected(self):
        """
        连接断开
        :return:
        """
        print(datetime.now(), '连接断开回调')

    def on_stock_order(self, order):
        """
        委托回报推送
        :param order: XtOrder对象
        :return:
        """
        order_str = "撤单成功" if order.order_status == xtconstant.ORDER_CANCELED else "挂单成功"
        print(
            f"""{order_str}！名称：{query_tool.get_name_by_code(tools.convert_stock_code(order.stock_code))}, 代码: {order.stock_code}, 委托类型: {order.order_type}, 委托数量: {order.order_volume}, 委托价格: {order.price}""")

    def on_stock_trade(self, trade):
        """
        成交变动推送
        :param trade: XtTrade对象
        :return:
        """
        print("on trade callback")
        # 判断是买入还是卖出
        trade_type = "BUY" if trade.offset_flag == xtconstant.OFFSET_FLAG_OPEN else "SELL"

        print(datetime.now(), '成交回调', trade.order_remark,
              f" ,委托方向: {'买入' if trade_type == 'BUY' else '卖出'} ,"
              f"成交价格 {trade.traded_price} 成交数量 {trade.traded_volume}")
        # 保存交易记录到数据库
        save_transaction_to_db(trade, trade_type)

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
        print(datetime.now(), sys._getframe().f_code.co_name)

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
        print(datetime.now(), sys._getframe().f_code.co_name)

    def on_account_status(self, status):
        """
        :param response: XtAccountStatus 对象
        :return:
        """
        print(datetime.now(), sys._getframe().f_code.co_name)


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


def get_ma5_price(stock_code, current_date=None, current_price=None):
    """获取指定股票的最新MA5价格，支持动态更新当日数据

    Args:
        stock_code (str): 股票代码（如'603722.SH'）
        current_date (datetime, optional): 当前日期（用于动态更新）
        current_price (float, optional): 当前价格（用于动态计算）

    Returns:
        float: 最新MA5价格（不足5日数据返回None）
    """
    pure_code = tools.convert_stock_code(stock_code)  # 如603722.SH -> sh603722
    df, _ = get_stock_data(pure_code)

    if df.empty or len(df) < 4:  # 至少需要4日历史数据
        print(f"警告：{stock_code} 数据不足，无法计算MA5")
        return None

    try:
        # 转换为时间序列索引
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 动态更新当日数据（如果传入参数）
        if current_date and current_price:
            # 确保日期格式一致
            current_date = pd.Timestamp(current_date)

            # 情况1：已有当日数据 → 更新收盘价
            if current_date in df.index:
                df.loc[df.index == current_date, 'close'] = current_price
            # 情况2：无当日数据 → 添加新行
            else:
                # 复制最近一日数据作为模板
                new_row = df.iloc[-1].copy()
                # 更新日期和收盘价
                new_row.name = current_date
                new_row['close'] = current_price
                # 添加到DataFrame
                df = pd.concat([df, new_row.to_frame().T])

        # 按时间升序排序（确保最新数据在最后）
        df = df.sort_index(ascending=True)

        # 核心计算逻辑[1,3,5](@ref)
        df['MA5'] = df['close'].rolling(
            window=5,
            min_periods=5  # 严格要求5个有效数据点
        ).mean().round(2)

        return df['MA5'].iloc[-1]  # 返回最新MA5值

    except KeyError as e:
        print(f"数据列缺失错误：{str(e)}")
        return None
    except Exception as e:
        print(f"MA5计算异常：{str(e)}")
        return None


def modify_last_days_and_calc_ma5(df, predict_ratio=1.04):
    """模拟预测MA5的核心方法（新增predict_ratio参数）"""
    if df.empty or len(df) < 2:
        raise ValueError("数据不足，至少需要2个交易日数据")

    modified_df = df.copy().sort_index(ascending=True)
    modified_df['adjusted_close'] = modified_df['close']

    # 复制最后一行并应用预测涨幅（原1.04倍逻辑）
    new_row = modified_df.iloc[-1].copy()
    new_row['close'] *= predict_ratio  # 动态传入预测系数
    new_row.name = new_row.name + pd.Timedelta(days=1)
    modified_df = pd.concat([modified_df, new_row.to_frame().T], axis=0)

    # 计算修正后的MA5
    modified_df['MA5'] = modified_df['close'].rolling(
        window=5, min_periods=1
    ).mean().round(2)
    return modified_df


def auto_order_by_ma5(stock_code, total_amount=12000):
    """瀑布流分层挂单策略"""
    # 获取基础MA5价格（不应用预测）
    base_ma5 = get_ma5_price(stock_code)
    if base_ma5 is None:
        return False

    # 分层配置（价格预测系数与金额比例）
    tiers = [
        {'ratio': 0.50, 'predict_ratio': 1.04},  # 第一档：预测1.04倍
        {'ratio': 0.25, 'predict_ratio': 1.025},  # 第二档：预测1.025倍
        {'ratio': 0.25, 'predict_ratio': 1.01}  # 第三档：预测1.01倍
    ]

    # 动态计算每层MA5预测价格
    tier_prices = []
    for tier in tiers:
        # 模拟不同预测倍数的MA5（需重新计算历史数据）
        df, _ = get_stock_data(tools.convert_stock_code(stock_code))
        if df.empty:
            continue

        modified_df = modify_last_days_and_calc_ma5(df, tier['predict_ratio'])
        tier_ma5 = modified_df['MA5'].iloc[-1]
        tier_prices.append({
            'price': round(tier_ma5, 2),
            'ratio': tier['ratio']
        })

    # 智能股数分配（严格100股整数倍）
    orders = []
    remaining_amount = min(total_amount, available_cash)

    # 第一轮：尝试三档分层
    for tier in tier_prices:
        if remaining_amount <= 0:
            break

        tier_max_amount = total_amount * tier['ratio']
        actual_amount = min(tier_max_amount, remaining_amount)

        # 计算可买股数（向下取整至100股）
        shares = int(actual_amount // (tier['price'] * 100)) * 100
        if shares == 0:
            continue  # 跳过无法成交的档位

        orders.append({'price': tier['price'], 'shares': shares})
        remaining_amount -= shares * tier['price']

    # 保底策略：若前三档未完成，合并为两档
    if len(orders) < 2 and remaining_amount > 0:
        backup_tiers = [
            {'predict_ratio': 1.04, 'ratio': 0.50},
            {'predict_ratio': 1.02, 'ratio': 0.50}
        ]
        tier_prices = []
        for tier in backup_tiers:
            df, _ = get_stock_data(tools.convert_stock_code(stock_code))
            modified_df = modify_last_days_and_calc_ma5(df, tier['predict_ratio'])
            tier_ma5 = modified_df['MA5'].iloc[-1]
            tier_prices.append({
                'price': round(tier_ma5, 2),
                'ratio': tier['ratio']
            })

        # 重新分配剩余资金
        remaining_amount = min(total_amount, available_cash)
        for tier in tier_prices:
            if remaining_amount <= 0:
                break

            tier_max_amount = total_amount * tier['ratio']
            actual_amount = min(tier_max_amount, remaining_amount)
            shares = int(actual_amount // (tier['price'] * 100)) * 100
            if shares > 0:
                orders.append({'price': tier['price'], 'shares': shares})
                remaining_amount -= shares * tier['price']

    # 执行挂单（需异步防阻塞）
    for order in orders:
        xt_trader.order_stock_async(
            acc, stock_code, xtconstant.STOCK_BUY,
            order['shares'], xtconstant.FIX_PRICE,
            order['price'], '瀑布流策略', stock_code
        )
        print(f"✅ 挂单成功：{order['shares']}股 @ {order['price']}（预测倍数：{order['price'] / base_ma5:.2f}x）")

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
            current_time = datetime.now()

            # 动态计算MA5（传入当前时间和价格）
            ma5_price = get_ma5_price(
                stock_code,
                current_date=current_time,
                current_price=current_price
            )
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


def sell_breached_stocks():
    """定时卖出所有跌破五日线的持仓"""
    try:
        # 获取当前时间
        now = datetime.now().strftime("%H:%M")
        print(f"\n=== 开始执行定时检测 ({now}) ===")

        # 检测跌破五日线股票
        breach_stocks = check_ma5_breach()
        if not breach_stocks:
            print("当前无持仓跌破五日线")
            return

        # 遍历卖出逻辑
        for stock in breach_stocks:
            stock_code = stock['代码']
            stock_name = stock['名称']
            hold_vol = stock['持有数量']

            # 获取实时可卖数量
            position = next((p for p in xt_trader.query_stock_positions(acc)
                             if p.stock_code == stock_code), None)
            if not position or position.m_nCanUseVolume <= 0:
                print(f"❌ {stock_name}({stock_code}) 无可卖持仓")
                continue

            # 获取实时行情数据（包含五档盘口）
            tick = xtdata.get_full_tick([stock_code])[stock_code]
            if not tick:
                print(f"⚠️ 无法获取 {stock_code} 实时行情")
                continue

            # 获取基准价格（最新成交价）
            base_price = tick['lastPrice']

            # 获取第五档买入价（买五价）或者现价跌1%的价格
            if 'bidPrice' in tick and len(tick['bidPrice']) >= 5:
                sell_price = tick['bidPrice'][4]  # 第五档买入价
            else:
                sell_price = base_price * 0.99  # 无五档数据时使用99%价格

            # 执行市价卖出
            async_seq = xt_trader.order_stock_async(
                acc,
                stock_code,
                xtconstant.STOCK_SELL,  # 卖出方向
                hold_vol,
                xtconstant.FIX_PRICE,  # 限价单模式
                sell_price,  # 计算的卖出价格
                'MA5止损策略',
                stock_code
            )
            print(f"✅ 已提交卖出订单：{stock_name}({stock_code}) {hold_vol}股")

    except Exception as e:
        print(f"‼️ 定时任务执行异常: {str(e)}")
    finally:
        print("=== 定时检测完成 ===\n")


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
    # 获取当前持仓股票集合
    positions = xt_trader.query_stock_positions(acc)
    hold_stocks = {pos.stock_code for pos in positions}
    # 取各品种 总持仓 可用持仓
    position_total_dict = {i.stock_code: i.m_nVolume for i in positions}
    position_available_dict = {i.stock_code: i.m_nCanUseVolume for i in positions}
    print(acc.account_id, '持仓字典', position_total_dict)
    print(acc.account_id, '可用持仓字典', position_available_dict)

    # 初始化日志记录器
    strategy_logger = setup_logger()
    strategy_logger.info("===== 策略启动 =====")

    # 创建并启动监控线程
    monitor_thread = threading.Thread(
        target=monitor_strategy_status,
        args=(strategy_logger,),
        daemon=True  # 设为守护线程，主程序退出时自动结束
    )
    monitor_thread.start()

    # breach_stocks = check_ma5_breach()
    # if breach_stocks:
    #     print("\n跌破五日线持仓预警（截至%s）" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    #     df = pd.DataFrame(breach_stocks)
    #     print(df[['代码', '名称', '持有数量', '当前价格', 'MA5价格']].to_string(index=False))
    # else:
    #     print("\n当前无持仓跌破五日线")

    # target_stocks = scan.get_target_stocks(False)
    # filtered_stocks = [code for code in target_stocks if code not in hold_stocks]
    # # 遍历过滤后的股票执行交易
    # for stock_code in filtered_stocks:
    #     # 动态二次校验（防止持仓变化）
    #     if stock_code in hold_stocks:
    #         continue
    #     success = auto_order_by_ma5(stock_code, 10000)
    #     if not success:
    #         print(f"【风控拦截】{stock_code} 下单失败，请检查数据完整性")

    scheduler = BackgroundScheduler(timezone="Asia/Shanghai")

    # 设置股票交易时间过滤（排除非交易日）
    scheduler.add_job(
        sell_breached_stocks,
        trigger=CronTrigger(
            hour=14,
            minute=53,
            day_of_week='mon-fri'  # 仅周一到周五
        ),
        misfire_grace_time=300  # 允许5分钟内的延迟执行
    )

    # 启动定时任务
    scheduler.start()
    print("定时任务已启动：每日14:56执行MA5止损检测")

    # tick = xtdata.get_full_tick(["603722.SH"])["603722.SH"]
    xtdata.run()
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
