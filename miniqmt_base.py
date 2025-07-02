import logging
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from xtquant import xtconstant
from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from miniqmt_data_utils import get_stock_data, get_ma5_price, modify_last_days_and_calc_ma5
from miniqmt_trade_utils import can_cancel_order_status, save_trigger_prices_to_csv, load_trigger_prices_from_csv
import first_limit_up_ma5_normal_scan as scan
import getAllStockCsv as tools
from miniqmt_callback import MyXtQuantTraderCallback
from position_manage.portfolio import Portfolio
from position_manage.transaction import Transaction
from miniqmt_logging_utils import setup_logger

query_tool = tools.StockQuery()
# ====== 全局策略配置 ======
PER_STOCK_TOTAL_BUDGET = 12000  # 每只股票的总买入预算 统一修改点
# 全局存储触发价格（格式：{股票代码: [触发价列表]})
trigger_prices = defaultdict(list)  # 使用 defaultdict 确保键不存在时自动创建空列表
# 在全局定义日志记录控制变量
log_throttle = defaultdict(lambda: {'last_log_time': 0, 'last_log_price': 0})

# 创建状态监控函数，每30分钟记录程序状态
def monitor_strategy_status(logger):
    while True:
        try:
            refresh_account_status()
            status_msg = (
                f"策略运行正常 | "
                f"可用资金: {available_cash:.2f} | "
                f"持仓数量: {len(hold_stocks)} | "
                f"总持仓: {position_total_dict}"
            )
            logger.info(status_msg)
            # 仅查询可撤委托
            orders = xt_trader.query_stock_orders(acc,cancelable_only=True)
            active_orders = [o for o in orders if can_cancel_order_status(o.order_status)]
            logger.info(f"活跃挂单数量: {len(active_orders)}")

        except Exception as e:
            logger.error(f"状态监控异常: {str(e)}")

        time.sleep(30 * 60)


def auto_order_by_ma5(stock_code, total_amount=10000):
    """瀑布流分层挂单策略"""
    base_ma5 = get_ma5_price(stock_code)
    if base_ma5 is None:
        return False

    # 分层配置（价格预测系数与金额比例）
    tiers = [
        {'predict_ratio': 1.04,'ratio': 0.50 },  # 第一档：预测1.04倍
        {'predict_ratio': 1.025,'ratio': 0.25 },  # 第二档：预测1.025倍
        {'predict_ratio': 1.01,'ratio': 0.25}  # 第三档：预测1.01倍
    ]

    # 动态计算每层MA5预测价格
    tier_prices = []
    for tier in tiers:
        # 模拟不同预测倍数的MA5（需重新计算历史数据）
        df, _ = get_stock_data(tools.convert_stock_code(stock_code), False)
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
            {'predict_ratio': 1.06, 'ratio': 0.50},
            {'predict_ratio': 1.03, 'ratio': 0.50}
        ]
        tier_prices = []
        for tier in backup_tiers:
            df, _ = get_stock_data(tools.convert_stock_code(stock_code), False)
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
        print(
            f"挂单成功：{order['shares']}股 @ {order['price']}（预算: {order['shares'] * order['price']:.2f}/{PER_STOCK_TOTAL_BUDGET}）")

    return True


def check_ma5_breach(positions,position_available_dict):
    """检测持仓中跌破五日线的股票"""
    breach_list = []
    for stock_code, hold_vol in position_available_dict.items():
        if hold_vol <= 0:
            continue

        try:
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

            if round(current_price, 2) < round(ma5_price, 2):
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
    """定时卖出所有跌破五日线的持仓及上一交易日涨停的股票"""
    try:
        now = datetime.now().strftime("%H:%M")
        print(f"\n=== 开始执行定时检测 ({now}) ===")

        positions = xt_trader.query_stock_positions(acc)
        # 检测跌破五日线的股票
        breach_stocks = check_ma5_breach(positions,position_available_dict)
        # # 检测上一交易日涨停且今日未涨停的股票
        # yesterday_limit_up_stocks = []
        # for pos in positions:
        #     if pos.m_nCanUseVolume <= 0:
        #         continue
        #
        #     stock_code = pos.stock_code
        #     try:
        #         # 使用get_stock_data获取T-1日数据（昨日）
        #         df_yesterday, _ = get_stock_data(tools.convert_stock_code(stock_code), False)
        #         if df_yesterday.empty or len(df_yesterday) < 2:
        #             continue
        #
        #         # 获取T-2日收盘价
        #         t2_close = df_yesterday['close'].iloc[-2]
        #         # 获取T-1日最高价和收盘价
        #         t1_high = df_yesterday['high'].iloc[-1]
        #         t1_close = df_yesterday['close'].iloc[-1]
        #
        #         # 计算T-1日涨停价（基于T-2日收盘价）
        #         t1_limit_up = round(t2_close * 1.1, 2)  # 主板10%涨停
        #         if stock_code.startswith('3') or stock_code.startswith('688'):  # 创业板/科创板20%
        #             t1_limit_up = round(t2_close * 1.2, 2)
        #
        #         # 判断T-1日是否涨停
        #         is_yesterday_limit_up = t1_high >= t1_limit_up - 0.01  # 考虑浮点误差
        #
        #         today_data = xtdata.get_market_data_ex(
        #             fields=['high'],
        #             stock_code=[stock_code],
        #             period='1d',
        #             count=1,
        #             subscribe=False
        #         )
        #
        #         if stock_code not in today_data or today_data[stock_code].empty:
        #             continue
        #
        #         today_high = today_data[stock_code]['high'].iloc[0]
        #         # 计算今日涨停价（基于T-1日收盘价）
        #         today_limit_up = round(t1_close * 1.1, 2)
        #         if stock_code.startswith('3') or stock_code.startswith('688'):
        #             today_limit_up = round(t1_close * 1.2, 2)
        #
        #         # 判断今日是否未涨停
        #         is_today_not_limit = today_high < today_limit_up - 0.01
        #
        #         # 合并判断条件
        #         if is_yesterday_limit_up and is_today_not_limit:
        #             stock_name = query_tool.get_name_by_code(stock_code)
        #             yesterday_limit_up_stocks.append({
        #                 '代码': stock_code,
        #                 '名称': stock_name,
        #                 '持有数量': pos.m_nCanUseVolume,
        #                 '类型': '上日涨停股'
        #             })
        #     except Exception as e:
        #         print(f"检测涨停股异常 {stock_code}: {str(e)}")
        #         continue

        # 合并卖出列表
        # all_sell_stocks = breach_stocks + yesterday_limit_up_stocks
        all_sell_stocks = breach_stocks
        if not all_sell_stocks:
            print("当前无符合卖出条件的持仓")
            return

        for stock in breach_stocks:
            stock_code = stock['代码']
            stock_name = stock['名称']
            hold_vol = stock['持有数量']
            position = next((p for p in xt_trader.query_stock_positions(acc)
                             if p.stock_code == stock_code), None)
            if not position or position.m_nCanUseVolume <= 0:
                print(f"{stock_name}({stock_code}) 无可卖持仓")
                continue
            # 获取实时行情数据（包含五档盘口）
            tick = xtdata.get_full_tick([stock_code])[stock_code]
            if not tick:
                print(f"⚠无法获取 {stock_code} 实时行情")
                continue
            # 确定卖出价格（第五档买入价或99%市价）
            base_price = tick['lastPrice']
            if 'bidPrice' in tick and len(tick['bidPrice']) >= 5:
                sell_price = tick['bidPrice'][4]
            else:
                sell_price = base_price * 0.99

            # 执行卖出
            async_seq = xt_trader.order_stock_async(
                acc,
                stock_code,
                xtconstant.STOCK_SELL,
                hold_vol,
                xtconstant.FIX_PRICE,
                sell_price,
                '涨停次日卖出策略' if '类型' in stock and stock['类型'] == '上日涨停股' else 'MA5止损策略',
                stock_code
            )
            reason = "上日涨停股" if '类型' in stock and stock['类型'] == '上日涨停股' else "跌破五日线"
            print(f"已提交卖出：{stock_name}({stock_code}) {hold_vol}股 | 原因：{reason}")

    except Exception as e:
        print(f"‼定时任务执行异常: {str(e)}")
    finally:
        print("=== 定时检测完成 ===\n")


def precompute_trigger_prices(stock_code):
    """预计算各层MA5触发价格"""
    base_ma5 = get_ma5_price(stock_code)
    if not base_ma5:
        print(f"⚠无法计算{stock_code}触发价: MA5数据缺失")
        return

    # 分层配置
    tiers = [
        {'predict_ratio': 1.06, 'weight': 0.4},
        {'predict_ratio': 1.03, 'weight': 0.3},
        {'predict_ratio': 1.01, 'weight': 0.3},
    ]

    # 生成触发价格
    for tier in tiers:
        df, _ = get_stock_data(tools.convert_stock_code(stock_code), False)
        modified_df = modify_last_days_and_calc_ma5(df, tier['predict_ratio'])
        trigger_price = round(modified_df['MA5'].iloc[-1], 2)

        # 计算预估挂单数量（100股整数倍）
        tier_budget = PER_STOCK_TOTAL_BUDGET * tier['weight']  # 层级预算
        estimated_shares = int(tier_budget // trigger_price) // 100 * 100  # 取整为100的倍数

        # 打印触发价格和预估挂单数量
        print(f"股票: {stock_code} | "
              f"层级触发价: {trigger_price:.2f} | "
              f"权重: {tier['weight'] * 100}% | "
              f"预估挂单: {estimated_shares}股")

        # 去重后存入全局变量
        if trigger_price not in trigger_prices[stock_code]:
            trigger_prices[stock_code].append({
                'price': trigger_price,
                'weight': tier['weight'],
                'triggered': False  # 触发标记
            })


def subscribe_target_stocks(target_stocks):
    for stock_code in target_stocks:
        # 预计算该股票的触发价格
        precompute_trigger_prices(stock_code)
        # 订阅分时数据（用于实时触发）
        xtdata.subscribe_quote(stock_code, period='tick', callback=on_quote_update)
        print(f"已订阅并计算触发价: {stock_code}")


def on_quote_update(data):
    try:
        current_time = time.time()
        if not hasattr(on_quote_update, 'logged_data_type'):
            strategy_logger.debug(f"行情数据结构: {type(data)}")
            on_quote_update.logged_data_type = True

        processed_stocks = []

        # 情况1：数据是字典格式（单股订阅）
        if isinstance(data, dict):
            for stock_code, quote in data.items():
                # 检查quote是字典还是列表
                if isinstance(quote, dict):
                    current_price = quote.get('lastPrice')
                elif isinstance(quote, list) and len(quote) > 0:
                    # 取列表中的最新行情
                    current_price = quote[0].get('lastPrice') if isinstance(quote[0], dict) else None
                else:
                    continue

                if current_price is not None:
                    processed_stocks.append(stock_code)
                    process_stock_quote(stock_code, current_price, current_time)

        # 情况2：数据是列表格式（多股订阅）
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    stock_code = item.get('stock_code', '')
                    current_price = item.get('lastPrice')

                    if stock_code and current_price is not None:
                        processed_stocks.append(stock_code)
                        process_stock_quote(stock_code, current_price, current_time)

        if processed_stocks:
            strategy_logger.debug(
                f"处理股票: {', '.join(processed_stocks[:3])}{'...' if len(processed_stocks) > 3 else ''}")

    except Exception as e:
        strategy_logger.error(f"行情处理异常: {str(e)}", exc_info=True)


def process_stock_quote(stock_code, current_price, current_time):
    """处理单只股票的行情更新"""
    # 1. 首次触发记录
    if stock_code not in log_throttle:
        print(f"开始监控 {stock_code} 实时行情，当前价格: {current_price}")
        log_throttle[stock_code] = {'last_log_time': current_time, 'last_log_price': current_price}

    # 2. 每60秒记录一次
    if current_time - log_throttle[stock_code]['last_log_time'] > 60:
        closest_tier = None
        min_diff = float('inf')
        for tier in trigger_prices.get(stock_code, []):
            if tier['triggered']:  # 跳过已触发的层级
                continue

            price_diff = abs(tier['price'] - current_price)
            if price_diff < min_diff:
                min_diff = price_diff
                closest_tier = tier

        if closest_tier:
            diff_percent = abs(closest_tier['price'] - current_price) / current_price * 100
            direction = "↑" if current_price < closest_tier['price'] else "↓"
            print(
                f"{stock_code} 行情: {current_price:.2f} | 最接近触发价: {closest_tier['price']} ({direction}{diff_percent:.2f}%)")
        else:
            print(f"{stock_code} 行情更新: {current_price:.2f} (无未触发价格层级)")

        log_throttle[stock_code]['last_log_time'] = current_time

    # 4. 触发条件检查
    for tier in trigger_prices.get(stock_code, []):
        if tier['triggered']:
            continue

        # 触发条件：当前价 <= 触发价（买入方向）
        if current_price <= tier['price']:
            strategy_logger.info(f"触发条件: {stock_code} 当前价 {current_price} ≤ 目标价 {tier['price']}")
            execute_trigger_order(stock_code, tier)
            tier['triggered'] = True


def execute_trigger_order(stock_code, tier):
    """执行触发挂单"""
    # 动态计算可用资金（每次触发时刷新）
    refresh_account_status()
    # 计算实际可买金额（不超过层级预算）
    available = min(PER_STOCK_TOTAL_BUDGET * tier['weight'], available_cash)

    # 计算可买数量（100股整数倍）
    buy_shares = int(available // (tier['price'] * 100)) * 100
    if buy_shares < 100:
        return

    # 记录触发时间
    tier['trigger_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    tier['triggered'] = True

    # 异步挂单（限价委托）
    xt_trader.order_stock_async(
        acc, stock_code, xtconstant.STOCK_BUY,
        buy_shares, xtconstant.FIX_PRICE,
        tier['price'], 'MA5触发策略', ''
    )
    print(f"⚡触发挂单：{stock_code} {buy_shares}股 @ {tier['price']}")

    # 立即保存更新后的触发状态
    save_trigger_prices_to_csv(trigger_prices)

def daily_pre_market_orders():
    """每日盘前挂单"""
    target_stocks = scan.get_target_stocks(False)
    filtered_stocks = [code for code in target_stocks if code not in hold_stocks]
    # 遍历过滤后的股票执行交易
    for stock_code in filtered_stocks:
        # 动态二次校验（防止持仓变化）
        if stock_code in hold_stocks:
            continue
        success = auto_order_by_ma5(stock_code, PER_STOCK_TOTAL_BUDGET)
        if not success:
            print(f"【风控拦截】{stock_code} 下单失败，请检查数据完整性")


def adjust_orders_at_935():
    """9:35定时任务：撤单后重新挂单，确保资金充分利用"""
    try:
        print("\n===== 9:35定时任务启动 =====")
        # 1. 撤掉所有未成交挂单
        cancel_all_pending_orders()
        # 2. 获取最新账户状态
        refresh_account_status()
        # 3. 获取目标股票列表
        target_stocks = scan.get_target_stocks(False)
        # 4. 过滤已持仓股票
        positions = xt_trader.query_stock_positions(acc)
        hold_stocks = {pos.stock_code for pos in positions}
        filtered_stocks = [code for code in target_stocks if code not in hold_stocks]

        if not filtered_stocks:
            print("⚠所有目标股票均已持仓，无需新增挂单")
            return

        # 5. 订阅并保存触发价格
        subscribe_target_stocks(filtered_stocks)
        save_trigger_prices_to_csv(trigger_prices)
        for code in filtered_stocks:
            if code not in trigger_prices or not trigger_prices[code]:
                print(f"警告: {code} 未生成触发价格层级")

    except Exception as e:
        print(f"‼ 9:35任务执行异常: {str(e)}")
    finally:
        print("===== 9:35定时任务完成 =====")


def analyze_trigger_performance(days=5):
    """分析最近N天的触发价格执行情况"""
    analysis_results = []

    for i in range(days):
        date_str = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        data = load_trigger_prices_from_csv(date_str)

        if data:
            total_triggers = sum(len(tiers) for tiers in data.values())
            triggered = sum(sum(1 for t in tiers if t['triggered']) for tiers in data.values())
            success_rate = triggered / total_triggers * 100 if total_triggers > 0 else 0

            analysis_results.append({
                'date': date_str,
                'stocks': len(data),
                'total_triggers': total_triggers,
                'triggered': triggered,
                'success_rate': f"{success_rate:.1f}%"
            })

    if analysis_results:
        df = pd.DataFrame(analysis_results)
        print("\n=== 触发价格执行情况分析 ===")
        print(df.to_string(index=False))
    else:
        print("⚠无历史数据可供分析")

def cancel_all_pending_orders():
    """撤掉所有未成交挂单"""
    orders = xt_trader.query_stock_orders(acc)
    if not orders:
        print("无待撤挂单")
        return

    success_count = 0
    for order in orders:
        if can_cancel_order_status(order.order_status):
            cancel_result = xt_trader.cancel_order_stock_async(acc, order.order_id)
            if cancel_result == 0:
                success_count += 1
                print(f"撤单成功：{order.stock_code} {order.order_volume}股")

    print(f"撤单完成：成功撤单 {success_count}/{len(orders)} 笔")


def refresh_account_status():
    """刷新账户状态"""
    global available_cash, hold_stocks, positions, position_total_dict, position_available_dict

    # 更新可用资金
    account_info = xt_trader.query_stock_asset(acc)
    available_cash = account_info.m_dCash

    # 更新持仓
    positions = xt_trader.query_stock_positions(acc)
    hold_stocks = {pos.stock_code for pos in positions}
    position_total_dict = {i.stock_code: i.m_nVolume for i in positions}
    position_available_dict = {i.stock_code: i.m_nCanUseVolume for i in positions}

    print(f"账户状态更新：可用资金={available_cash:.2f}, 持仓数量={len(hold_stocks)}")


if __name__ == "__main__":
    xtdata.enable_hello = False
    path = r'D:\备份\国金证券QMT交易端\userdata_mini'
    session_id = int(time.time())
    xt_trader = XtQuantTrader(path, session_id)

    acc = StockAccount('8886969255', 'STOCK')
    # 创建交易回调类对象，并声明接收回调
    callback = MyXtQuantTraderCallback(query_tool)
    xt_trader.register_callback(callback)
    # 启动交易线程
    xt_trader.start()
    # 建立交易连接，返回0表示连接成功
    connect_result = xt_trader.connect()
    print('建立交易连接，返回0表示连接成功', connect_result)
    # 对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功
    subscribe_result = xt_trader.subscribe(acc)
    print('对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功', subscribe_result)

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
    scheduler = BackgroundScheduler(timezone="Asia/Shanghai")

    # 获取当前持仓股票集合
    positions = xt_trader.query_stock_positions(acc)
    hold_stocks = {pos.stock_code for pos in positions}

    scheduler.add_job(
        daily_pre_market_orders,
        trigger=CronTrigger(
            hour=9,
            minute=15,
            day_of_week='mon-fri'
        ),
        misfire_grace_time=60
    )
    print("定时任务已添加：每日9:15执行盘前挂单")

    scheduler.add_job(
        adjust_orders_at_935,
        trigger=CronTrigger(
            hour=9,
            minute=35,
            day_of_week='mon-fri'
        ),
        misfire_grace_time=60
    )
    print("定时任务已添加：每日9:35执行订单调整")


    # 检查当前时间并立即执行
    def check_and_execute():
        now = datetime.now()
        start_time = now.replace(hour=9, minute=35, second=0, microsecond=0)
        end_time = now.replace(hour=14, minute=50, second=0, microsecond=0)

        if start_time <= now <= end_time and now.weekday() < 5:  # 0-4 表示周一到周五
            print("🕒 当前时间在 9:35-14:50 之间，立即执行订单调整")
            adjust_orders_at_935()


    # 启动前执行检查
    check_and_execute()

    scheduler.add_job(
        sell_breached_stocks,
        trigger=CronTrigger(
            hour=14,
            minute=54,
            day_of_week='mon-fri'
        ),
        misfire_grace_time=60
    )
    print("定时任务已启动：每日14:54执行MA5止损检测")

    scheduler.add_job(
        analyze_trigger_performance,
        trigger=CronTrigger(
            hour=15,
            minute=5,
            day_of_week='mon-fri'
        )
    )
    print("定时任务已添加：每日15:05执行触发价格分析")

    scheduler.start()
    xtdata.run()
