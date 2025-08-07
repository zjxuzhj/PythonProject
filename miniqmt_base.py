import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from xtquant import xtconstant
from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount

import first_limit_up_ma5_normal as normal
import first_limit_up_ma5_normal_scan as scan
import getAllStockCsv as tools
from common_sell_logic import get_sell_decision, MarketDataContext
from first_limit_up_ma5_normal import StrategyConfig
from miniqmt_callback import MyXtQuantTraderCallback
from miniqmt_data_utils import get_stock_data, get_ma5_price, modify_last_days_and_calc_ma5
from miniqmt_logging_utils import setup_logger
from miniqmt_trade_utils import can_cancel_order_status, save_trigger_prices_to_csv, load_trigger_prices_from_csv

query_tool = tools.StockQuery()
# ====== 全局策略配置 ======
PER_STOCK_TOTAL_BUDGET = 20000  # 每只股票的总买入预算
PER_FOURTH_STOCK_TOTAL_BUDGET = PER_STOCK_TOTAL_BUDGET  # 涨停后第四天的每只股票的总买入预算(暂时改为统一，因为第四天并没有特别优势)
daily_fourth_day_stocks = set()  # 存储当天的第四天股票列表
# 全局存储触发价格（格式：{股票代码: [触发价列表]})
trigger_prices = defaultdict(list)  # 使用 defaultdict 确保键不存在时自动创建空列表
# 在全局定义日志记录控制变量
log_throttle = defaultdict(lambda: {'last_log_time': 0, 'last_log_price': 0})
# ====== 全局风险偏好配置 ======
RISK_LEVEL = 'low'  # 可选项：'high', 'medium', 'low'


# ====== 风险偏好设置函数 ======
def set_risk_level(level):
    """设置全局风险偏好"""
    global RISK_LEVEL
    valid_levels = ['high', 'medium', 'low']

    if level.lower() in valid_levels:
        RISK_LEVEL = level.lower()
        print(f"✅ 风险偏好已更新为: {RISK_LEVEL.upper()}")
    else:
        print(f"⚠ 无效风险等级: {level}。有效选项: {', '.join(valid_levels)}")


# ====== 分层策略配置 ======
def get_tiers_by_risk_level():
    """根据风险偏好返回对应的分层配置"""
    if RISK_LEVEL == 'high':
        return [
            {'predict_ratio': 1.06, 'ratio': 0.50},
            {'predict_ratio': 1.04, 'ratio': 0.25},
            {'predict_ratio': 1.02, 'ratio': 0.25}
        ]
    elif RISK_LEVEL == 'low':
        return [
            {'predict_ratio': 1.04, 'ratio': 1},
            {'predict_ratio': 1.04, 'ratio': 0.35},
            {'predict_ratio': 1.04, 'ratio': 0.25}
        ]
    else:  # 默认中风险
        return [
            {'predict_ratio': 1.04, 'ratio': 0.50},
            {'predict_ratio': 1.025, 'ratio': 0.25},
            {'predict_ratio': 1.01, 'ratio': 0.25}
        ]


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
            orders = xt_trader.query_stock_orders(acc, cancelable_only=True)
            active_orders = [o for o in orders if can_cancel_order_status(o.order_status)]
            logger.info(f"活跃挂单数量: {len(active_orders)}")

        except Exception as e:
            logger.error(f"状态监控异常: {str(e)}")

        time.sleep(30 * 60)


def check_ma5_breach(positions, position_available_dict):
    """检测持仓中跌破五日线的股票"""
    breach_list = []
    for stock_code, hold_vol in position_available_dict.items():
        if hold_vol <= 0:
            continue

        try:
            tick = xtdata.get_full_tick([stock_code])[stock_code]
            current_price = tick['lastPrice']
            if current_price is None or current_price <= 0:
                print(f"无法获取 {stock_code} 的有效实时价格，跳过本次检测。")
                continue
            current_time = datetime.now()

            # 动态计算MA5（传入当前时间和价格）
            ma5_price = get_ma5_price(
                stock_code,
                current_date=current_time,
                current_price=current_price
            )
            if ma5_price is None or ma5_price == 0:
                continue

            deviation = (current_price - ma5_price) / ma5_price
            if deviation <= -0.004:  # 跌幅超过五日线的0.4%卖出
                stock_name = query_tool.get_name_by_code(stock_code)
                breach_list.append({
                    '代码': stock_code,
                    '名称': stock_name,
                    '持有数量': hold_vol,
                    '当前价格': current_price,
                    'MA5价格': ma5_price,
                })
        except Exception as e:
            print(f"检测异常 {stock_code}: {str(e)}")
            continue
    return breach_list


def sell_breached_stocks():
    """定时卖出所有符合卖出条件的持仓股票"""
    try:
        now_dt = datetime.now()
        now_str = now_dt.strftime("%H:%M")
        print(f"\n=== 开始执行通用定时卖出检测 ({now_str}) ===")

        positions = xt_trader.query_stock_positions(acc)
        if not positions:
            print("当前无持仓。")
            return
        sell_list = []
        config = StrategyConfig()
        # 统一遍历所有持仓
        for pos in positions:
            if pos.m_nCanUseVolume <= 0:
                continue
            stock_code = pos.stock_code
            try:
                tick = xtdata.get_full_tick([stock_code])[stock_code]
                hist_df, _ = get_stock_data(tools.convert_stock_code(stock_code), False)
                if tick is None or hist_df.empty or len(hist_df) < 2:
                    print(f"数据不足，跳过 {stock_code} 的卖出检测。")
                    continue
                position_info = {'hold_days': 2}

                ma5_price = get_ma5_price(stock_code, current_date=now_dt, current_price=tick['lastPrice'])
                if ma5_price is None: continue

                limit_rate = config.MARKET_LIMIT_RATES[normal.get_market_type(stock_code)]

                high_price = tick.get('high')  # 今天的最高价
                last_close = tick.get('lastClose')  # 昨天的收盘价
                current_price = tick.get('lastPrice')  # 最新成交价
                today_up_limit_price = round(last_close * (1 + limit_rate), 2)
                today_down_limit_price = round(last_close * (1 - limit_rate), 2)

                # iloc[-1]：获取 DataFrame最后一行（最新交易日，记为T-1 日）
                t1_close = hist_df['close'].iloc[-1]
                t2_close = hist_df['close'].iloc[-2]
                t1_limit_price = round(t2_close * (1 + limit_rate), 2)
                t1_down_limit_price = round(t2_close * (1 - limit_rate), 2)
                market_data = MarketDataContext(
                    high=high_price,
                    close=current_price,
                    ma5=ma5_price,
                    up_limit_price=today_up_limit_price,
                    down_limit_price=today_down_limit_price,
                    prev_close=t1_close,
                    prev_up_limit_price=t1_limit_price,
                    prev_down_limit_price=t1_down_limit_price
                )
                should_sell, reason = get_sell_decision(position_info, market_data)

                if should_sell:
                    stock_name = query_tool.get_name_by_code(stock_code)
                    sell_list.append({
                        '代码': stock_code,
                        '名称': stock_name,
                        '持有数量': pos.m_nCanUseVolume,
                        '原因': reason
                    })
            except Exception as e:
             print(f"检测 {stock_code} 异常: {e}")
             continue

        # 统一执行卖出
        if not sell_list:
            print("当前无符合卖出条件的持仓。")
            return

        print("--- 统一决策后待卖出列表 ---")
        for stock in sell_list:
            print(f"  - 代码: {stock['代码']}, 名称: {stock['名称']}, 数量: {stock['持有数量']}, 原因: {stock['原因']}")
        print("--------------------------")

        for stock in sell_list:
            stock_code = stock['代码']
            stock_name = stock['名称']
            hold_vol = stock['持有数量']
            position = next((p for p in xt_trader.query_stock_positions(acc)
                             if p.stock_code == stock_code), None)
            if not position or position.m_nCanUseVolume <= 0:
                print(f"{stock_name}({stock_code}) 无可卖持仓，跳过。")
                continue
            # 获取实时行情数据（包含五档盘口）
            tick = xtdata.get_full_tick([stock_code])[stock_code]
            if not tick:
                print(f"⚠无法获取 {stock_code} 实时行情，跳过。")
                continue
            # 确定卖出价格（第五档买入价或99%市价）
            base_price = tick['lastPrice']
            if 'bidPrice' in tick and len(tick['bidPrice']) >= 5:
                sell_price = tick['bidPrice'][4]
            else:
                sell_price = base_price * 0.99

            # 执行卖出
            reason_memo = "涨停次日卖出策略" if '类型' in stock and stock['类型'] == '涨停断板' else "MA5止损策略"
            async_seq = xt_trader.order_stock_async(
                acc,
                stock_code,
                xtconstant.STOCK_SELL,
                hold_vol,
                xtconstant.FIX_PRICE,
                sell_price,
                reason_memo,
                stock_code
            )
            reason = "涨停断板" if '类型' in stock and stock['类型'] == '涨停断板' else "跌破五日线"
            print(f"已提交卖出：{stock_name}({stock_code}) {hold_vol}股 | 原因：{reason}")

    except Exception as e:
        print(f"‼定时任务执行异常: {str(e)}")
    finally:
        print("=== 定时检测完成 ===\n")


def precompute_trigger_prices(stock_code):
    # 确定股票类型和对应预算
    if stock_code in daily_fourth_day_stocks:
        total_budget = PER_FOURTH_STOCK_TOTAL_BUDGET
        print(f"⭐ {stock_code} 是第四天股票，使用专用预算")
    else:
        total_budget = PER_STOCK_TOTAL_BUDGET
    """预计算各层MA5触发价格"""
    current_tiers = get_tiers_by_risk_level()
    if not current_tiers:  # 检查层级数据是否为空
        print(f"⚠ {stock_code} 无可用层级数据")
        return
    base_ma5 = get_ma5_price(stock_code)
    if not base_ma5:
        print(f"⚠无法计算{stock_code}触发价: MA5数据缺失")
        return

    # 生成触发价格，现在只生成一档
    for i, tier in enumerate(current_tiers):
        if i > 0:
            break
        df, _ = get_stock_data(tools.convert_stock_code(stock_code), False)
        modified_df = modify_last_days_and_calc_ma5(df, tier['predict_ratio'])
        trigger_price = round(modified_df['MA5'].iloc[-1], 2)

        # 计算预估挂单数量（100股整数倍）
        tier_budget = total_budget * tier['ratio']  # 层级预算
        estimated_shares = int(tier_budget // trigger_price) // 100 * 100  # 取整为100的倍数

        # 打印触发价格和预估挂单数量
        print(f"股票: {stock_code} | "
              f"层级触发价: {trigger_price:.2f} | "
              f"权重: {tier['ratio'] * 100}% | "
              f"预估挂单: {estimated_shares}股")

        # 去重后存入全局变量
        if trigger_price not in trigger_prices[stock_code]:
            trigger_prices[stock_code].append({
                'price': trigger_price,
                'ratio': tier['ratio'],
                'triggered': False  # 触发标记
            })


def subscribe_target_stocks(filtered_stocks):
    for stock_code in filtered_stocks:
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

                if current_price is not None and current_price > 0:
                    processed_stocks.append(stock_code)
                    process_stock_quote(stock_code, current_price, current_time)
                else:
                    strategy_logger.debug(f"接收到 {stock_code} 的无效价格: {current_price}")

        # 情况2：数据是列表格式（多股订阅）
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    stock_code = item.get('stock_code', '')
                    current_price = item.get('lastPrice')

                    if stock_code and current_price is not None and current_price > 0:
                        processed_stocks.append(stock_code)
                        process_stock_quote(stock_code, current_price, current_time)
                    else:
                        strategy_logger.debug(f"接收到 {stock_code} 的无效价格: {current_price}")

        if processed_stocks:
            strategy_logger.debug(
                f"处理股票: {', '.join(processed_stocks[:3])}{'...' if len(processed_stocks) > 3 else ''}")

    except Exception as e:
        strategy_logger.error(f"行情处理异常: {str(e)}", exc_info=True)


def process_stock_quote(stock_code, current_price, current_time):
    """处理单只股票的行情更新"""
    # ===== 增加价格有效性校验，作为双重保险 =====
    if current_price is None or current_price <= 0:
        # 记录一次警告即可，防止日志刷屏
        if stock_code not in log_throttle or log_throttle[stock_code].get('last_log_price', 0) > 0:
            strategy_logger.warning(f"跳过对 {stock_code} 的处理，因接收到无效价格: {current_price}")
        return  # 直接返回，不执行后续逻辑
    stock_name = query_tool.get_name_by_code(stock_code)
    name_display = f"{stock_name}{'  ' if len(stock_name) == 3 else ''}"
    # 1. 首次触发记录
    if stock_code not in log_throttle:
        print(f"开始监控 {name_display} 实时行情，当前价格: {current_price:.2f}")
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
                f"{name_display} 行情: {current_price:.2f} | 最接近触发价: {closest_tier['price']} ({direction}{diff_percent:.2f}%)")
        else:
            print(f"{name_display} 行情更新: {current_price:.2f} (无未触发价格层级)")

        log_throttle[stock_code]['last_log_time'] = current_time

    # 4. 触发条件检查
    for tier in trigger_prices.get(stock_code, []):
        if tier['triggered']:
            continue

        # 触发条件：当前价 <= 触发价（买入方向）
        if current_price <= tier['price']:
            strategy_logger.info(f"触发条件: {name_display} 当前价 {current_price} ≤ 目标价 {tier['price']}")
            execute_trigger_order(name_display, stock_code, tier)
            tier['triggered'] = True


def execute_trigger_order(name_display, stock_code, tier):
    """执行触发挂单"""
    # 确定股票类型和对应预算
    if stock_code in daily_fourth_day_stocks:
        total_budget = PER_FOURTH_STOCK_TOTAL_BUDGET
        prefix = "⚡⚡⚡"  # 特殊标记
    else:
        total_budget = PER_STOCK_TOTAL_BUDGET
        prefix = "⚡"
    # 动态计算可用资金（每次触发时刷新）
    refresh_account_status()
    # 计算实际可买金额（不超过层级预算）
    available = min(total_budget * tier['ratio'], available_cash)

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
    print(f"{prefix}触发挂单：{name_display} {buy_shares}股 @ {tier['price']}")

    # 立即保存更新后的触发状态
    save_trigger_prices_to_csv(trigger_prices)


def daily_pre_market_orders():
    """
        暂时移除盘前挂单，因为和935的策略产生冲突
        每日盘前“广度优先”挂单策略。
        为所有目标股票计算分层价格，然后只挂出每个股票的第一档买单。
        """
    global daily_fourth_day_stocks
    try:
        now = datetime.now().strftime("%H:%M")
        print(f"\n===== 开始执行盘前广度优先挂单 ({now}) =====")

        # 1. 刷新账户状态，获取最新的可用资金和持仓
        refresh_account_status()

        # 2. 获取目标股票列表，并过滤掉已持仓的
        target_stocks, fourth_day_stocks = scan.get_target_stocks(False)
        daily_fourth_day_stocks = set(fourth_day_stocks)
        filtered_stocks = [code for code in target_stocks if code not in hold_stocks]

        if not filtered_stocks:
            print("所有目标股票均已持仓或无新目标，无需挂单。")
            return

        print(f"筛选后得到 {len(filtered_stocks)} 个目标股票: {filtered_stocks}")

        # 3. 清空旧的触发价格，为当天计算做准备
        trigger_prices.clear()

        # 4. 为所有目标股票预计算并填充所有层级的触发价格到全局变量
        for stock_code in filtered_stocks:
            precompute_trigger_prices(stock_code)

        # 5. 将今天所有计划的触发价格（所有档位）保存到CSV，作为今日交易计划的备份
        if trigger_prices:
            save_trigger_prices_to_csv(trigger_prices)
            print("今日所有目标股票的触发价格已计算并保存。")
        else:
            print("⚠ 未能为任何股票生成触发价格，盘前挂单终止。")
            return

        # 6. 遍历所有股票，只挂出第一档买单
        # 使用一个局部变量来追踪模拟挂单后的可用资金
        available_cash_tracker = available_cash
        print(f"开始挂单... 初始可用资金: {available_cash_tracker:.2f}")

        for stock_code, tiers in trigger_prices.items():
            if not tiers:
                print(f"⚠ {stock_code} 无有效触发价格层级，跳过。")
                continue

            # 获取第一档（最高优先级）的触发信息
            first_tier = tiers[0]
            order_price = first_tier['price']

            # 根据第一档的预算比例计算挂单金额和股数
            tier_budget = PER_STOCK_TOTAL_BUDGET * first_tier['ratio']
            order_shares = int(tier_budget // (order_price * 100)) * 100

            if order_shares < 100:
                print(f"跳过 {stock_code}: 根据预算计算股数不足100股。")
                continue

            order_cost = order_price * order_shares

            # 检查模拟资金是否足够
            if available_cash_tracker >= order_cost:
                # 资金充足，执行异步挂单
                xt_trader.order_stock_async(
                    acc,
                    stock_code,
                    xtconstant.STOCK_BUY,
                    order_shares,
                    xtconstant.FIX_PRICE,
                    order_price,
                    '广度优先-第一档',  # 使用清晰的备注
                    stock_code
                )
                # 更新模拟可用资金
                available_cash_tracker -= order_cost
                stock_name = query_tool.get_name_by_code(stock_code)
                print(
                    f"✅ 已提交第一档买单: {stock_name}({stock_code}) {order_shares}股 @ {order_price:.2f} | 预估成本: {order_cost:.2f}")
                print(f"   剩余模拟资金: {available_cash_tracker:.2f}")

            else:
                print(
                    f"资金不足，无法为 {stock_code} 挂单。所需资金: {order_cost:.2f}, 剩余: {available_cash_tracker:.2f}")
                # 后续的股票也不再尝试，因为资金已经不足
                break

    except Exception as e:
        print(f"‼ 盘前广度优先挂单任务执行异常: {str(e)}")
    finally:
        print("===== 盘前广度优先挂单完成 =====\n")


def adjust_orders_at_910():
    global daily_fourth_day_stocks
    try:
        now = datetime.now().strftime("%H:%M")
        print(f"\n===== 开始执行910策略 ({now}) =====")
        # 获取最新账户状态
        refresh_account_status()

        filtered_stocks = []
        if trigger_prices:
            print("内存中存在触发价格数据，直接使用")
            all_codes = list(set(trigger_prices.keys()))
            for code in all_codes:
                filtered_stocks.append(code)
        else:
            target_stocks, fourth_day_stocks = scan.get_target_stocks(False)
            daily_fourth_day_stocks = set(fourth_day_stocks)
            # 过滤已持仓股票
            positions = xt_trader.query_stock_positions(acc)
            hold_stocks = {pos.stock_code for pos in positions}
            file_data = load_trigger_prices_from_csv()
            if file_data:
                trigger_prices.update(file_data)
                # 动态过滤：只处理有未触发档位的股票
                for code in target_stocks:
                    # 检查触发记录
                    if code in trigger_prices:
                        untriggered_tiers = [t for t in trigger_prices[code] if not t['triggered']]
                        # 如果存在未触发的层级，加入筛选结果
                        if untriggered_tiers:
                            filtered_stocks.append(code)
                    # 无记录时视为新股票
                    else:
                        if code not in hold_stocks:
                            filtered_stocks.append(code)
            else:
                for code in target_stocks:
                    if code not in hold_stocks:
                        filtered_stocks.append(code)
                # 没有csv数据时单独计算每个股的档位价格
                for stock_code in filtered_stocks:
                    precompute_trigger_prices(stock_code)
                save_trigger_prices_to_csv(trigger_prices)

        if not filtered_stocks:
            print("⚠所有目标股票均已持仓，无需新增挂单")
            return

        # 保存更新后的触发价格
        save_trigger_prices_to_csv(trigger_prices, True)

        # 订阅价格监控
        subscribe_target_stocks(filtered_stocks)

        # 验证触发价格数据完整性
        valid_stocks = []
        for code in filtered_stocks:
            if code not in trigger_prices or not trigger_prices[code]:
                print(f"警告: {code} 未生成触发价格层级")
            else:
                valid_stocks.append(code)
        if not valid_stocks:
            print("⚠ 所有目标股票均缺少有效触发价格数据")
            return

    except Exception as e:
        print(f"‼ 9:10任务执行异常: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("===== 9:10定时任务完成 =====")


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
    set_risk_level('low')
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

    adjust_orders_at_910()

    scheduler.add_job(
        sell_breached_stocks,
        trigger=CronTrigger(
            hour=14,
            minute=56,
            day_of_week='mon-fri'
        ),
        misfire_grace_time=60
    )
    print("定时任务已启动：每日14:54执行MA5止损检测")

    scheduler.add_job(
        analyze_trigger_performance,
        trigger=CronTrigger(
            hour=15,
            minute=1,
            day_of_week='mon-fri'
        )
    )
    print("定时任务已添加：每日15:01执行触发价格分析")

    scheduler.start()
    xtdata.run()
