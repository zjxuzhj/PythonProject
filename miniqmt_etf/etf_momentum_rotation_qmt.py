# ETF动量轮动策略 - MiniQMT版本 (实盘强一致性版)
# 核心逻辑：11:00触发 -> 计算动量 -> 卖出非目标 -> 阻塞等待成交 -> 资金同步 -> 买入目标
# 确保文件编码为 UTF-8

import threading
import time
import signal
import sys
import os
import traceback
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from xtquant import xtconstant, xtdata
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount

# ====== 路径设置 (保持原有结构) ======
import sys as _sys, os as _os

_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

# 导入自定义模块
import getAllStockCsv as tools
from data_ingestion import updateAllStockDataCache as data_updater
from miniqmt_etf.etf_momentum_core import ETFMomentumCore
from miniqmt_etf.miniqmt_callback import MyXtQuantTraderCallback
from miniqmt_etf.miniqmt_logging_utils import setup_logger

# ====== 全局配置 ======
_env_alloc = os.environ.get('ALLOCATED_CAPITAL')
TOTAL_BUDGET = float(_env_alloc) if _env_alloc else 5000  # 总预算(仅作参考,实盘读取账户资金)
MIN_TRADE_AMOUNT = 1000  # 最小交易金额
MOMENTUM_DAYS = 25  # 动量计算窗口
MAX_ETF_COUNT = 1  # 持仓数量限制

# 全局变量初始化
query_tool = tools.StockQuery()
strategy_logger = None
xt_trader = None
acc = None


class ETFMomentumStrategy:
    """ETF动量轮动策略类 - 实盘原子化交易版"""

    def __init__(self, trader, account, logger):
        self.trader = trader
        self.account = account
        self.logger = logger
        self.core = ETFMomentumCore(MOMENTUM_DAYS)
        # 环境变量覆盖最大持仓数
        try:
            env_max = os.environ.get('MAX_POSITIONS')
            self.max_etf_count = int(env_max) if env_max else MAX_ETF_COUNT
        except Exception:
            self.max_etf_count = MAX_ETF_COUNT

    def filter_etfs(self):
        """筛选ETF并计算得分"""
        try:
            self.logger.info("正在计算ETF动量得分...")
            # 获取实时得分
            etf_scores = self.core.get_etf_scores_for_live()
            # 打印得分表
            self.core.print_etf_scores(etf_scores)

            # 过滤有效得分的ETF
            valid_etfs = [data for data in etf_scores if 0 < data['score'] < 6]

            # 返回排名前N的目标
            return valid_etfs[:self.max_etf_count]
        except Exception as e:
            self.logger.error(f"筛选ETF异常: {str(e)}")
            return []

    def get_current_positions(self):
        """获取当前账户持仓详情"""
        try:
            positions = self.trader.query_stock_positions(self.account)
            position_dict = {}
            for pos in positions:
                if pos.volume > 0:
                    position_dict[pos.stock_code] = {
                        'volume': pos.volume,
                        'can_use_volume': pos.can_use_volume,
                        'market_value': pos.market_value
                    }
            return position_dict
        except Exception as e:
            self.logger.error(f"获取持仓异常: {str(e)}")
            return {}

    def get_available_cash(self):
        """获取账户可用资金"""
        try:
            asset = self.trader.query_stock_asset(self.account)
            if asset:
                return asset.cash
            return 0.0
        except Exception as e:
            self.logger.error(f"获取可用资金异常: {str(e)}")
            return 0.0

    def place_order(self, stock_code, action_type, volume):
        """
        核心下单函数
        使用对手盘五档价格(买入用卖五, 卖出用买五)确保成交
        :return: order_seq (int) 或 -1
        """
        try:
            # 获取五档行情
            tick_data = xtdata.get_full_tick([stock_code])
            if not tick_data or stock_code not in tick_data:
                self.logger.warning(f"{stock_code} 获取行情失败，无法下单")
                return -1

            tick = tick_data[stock_code]
            price = 0.0

            # 价格策略：激进成交
            if action_type == xtconstant.STOCK_BUY:
                asks = tick.get('askPrice', [])
                # 如果有卖五价就用卖五，否则用最新价
                price = asks[4] if len(asks) >= 5 else tick.get('lastPrice', 0)
            elif action_type == xtconstant.STOCK_SELL:
                bids = tick.get('bidPrice', [])
                # 如果有买五价就用买五，否则用最新价
                price = bids[4] if len(bids) >= 5 else tick.get('lastPrice', 0)

            if price <= 0:
                self.logger.warning(f"{stock_code} 价格无效({price})，取消下单")
                return -1

            # 记录日志
            action_str = "买入" if action_type == xtconstant.STOCK_BUY else "卖出"
            self.logger.info(f"发起{action_str}: {stock_code}, 数量:{volume}, 委托价:{price:.3f}")

            # 异步下单，返回序列号
            seq = self.trader.order_stock_async(
                self.account, stock_code, action_type, int(volume),
                xtconstant.FIX_PRICE, price, 'ETF动量轮动', ''
            )
            return seq

        except Exception as e:
            self.logger.error(f"下单执行异常: {e}")
            return -1

    def wait_for_order_filled(self, order_seq, timeout=45):
        """
        阻塞直到订单成交
        :param order_seq: 订单序列号
        :return: Boolean (是否成功成交)
        """
        if order_seq == -1: return False

        start_time = time.time()
        self.logger.info(f"等待订单(Seq:{order_seq})成交...")

        while time.time() - start_time < timeout:
            orders = self.trader.query_stock_orders(self.account)
            target_order = None

            # 查找对应订单
            for o in orders:
                if o.order_seq == order_seq:
                    target_order = o
                    break

            if target_order:
                # 状态码 50: 全部成交
                if target_order.order_status == 50:
                    self.logger.info(f"订单(Seq:{order_seq}) 已全部成交。")
                    return True
                # 状态码 52: 废单, 53: 撤单, 54: 撤单
                elif target_order.order_status in [52, 53, 54]:
                    self.logger.error(
                        f"订单(Seq:{order_seq}) 异常结束，状态码: {target_order.order_status}, 备注: {target_order.order_remark}")
                    return False

            time.sleep(1)  # 每秒轮询一次

        self.logger.warning(f"订单(Seq:{order_seq}) 等待超时({timeout}s)，未确认完全成交。")
        return False

    def execute_rebalance(self):
        """
        【原子化调仓主逻辑】
        1. 计算目标
        2. 卖出非目标 -> 等待成交
        3. 资金同步缓冲
        4. 全仓买入目标 -> 等待成交
        """
        self.logger.info("========== [11:00] 开始执行动量轮动调仓 ==========")

        # 1. 计算目标
        target_etfs = self.filter_etfs()
        if not target_etfs:
            self.logger.warning("未筛选出有效目标ETF，停止调仓。")
            return

        target_code = target_etfs[0]['stock_code']
        target_name = self.core.etf_names.get(target_code, '未知')
        self.logger.info(f"今日策略目标: {target_name} ({target_code})")

        # 2. 获取持仓并处理卖出
        positions = self.get_current_positions()
        strategy_pool = self.core.ETF_POOL  # 只操作策略池内的标的

        has_sold = False

        for stock_code, pos_data in positions.items():
            # 如果持仓在策略池中，且不是今日目标 -> 卖出
            if stock_code in strategy_pool and stock_code != target_code:
                can_sell_vol = pos_data['can_use_volume']
                if can_sell_vol > 0:
                    stock_name = self.core.etf_names.get(stock_code, '未知')
                    self.logger.info(f"持仓需切换，准备卖出: {stock_name}({stock_code})")

                    # 发出卖单
                    seq = self.place_order(stock_code, xtconstant.STOCK_SELL, can_sell_vol)

                    # 阻塞等待成交
                    is_filled = self.wait_for_order_filled(seq, timeout=45)
                    if is_filled:
                        has_sold = True
                    else:
                        self.logger.error(f"卖出 {stock_code} 失败或超时，买入步骤可能因资金不足受阻。")

        # 3. 资金同步等待
        if has_sold:
            self.logger.info("卖出成交确认，等待5秒以同步账户资金状态...")
            time.sleep(5)

        # 4. 执行买入
        # 重新检查持仓（防止重复买入）
        positions_now = self.get_current_positions()
        if target_code in positions_now:
            self.logger.info(f"已持有目标 {target_name}，无需买入。")
            self.logger.info("========== 调仓任务完成 ==========")
            return

        # 获取可用资金
        available_cash = self.get_available_cash()
        self.logger.info(f"当前可用资金: {available_cash:.2f}")

        if available_cash < MIN_TRADE_AMOUNT:
            self.logger.warning("可用资金低于最小交易限额，放弃买入。")
            return

        # 获取买入价格参考
        tick = xtdata.get_full_tick([target_code])
        if target_code not in tick:
            self.logger.error(f"无法获取 {target_code} 行情，放弃买入")
            return

        ask_prices = tick[target_code].get('askPrice', [])
        # 使用卖五价作为成本预估，防止价格波动导致资金不足
        buy_ref_price = ask_prices[4] if len(ask_prices) >= 5 else tick[target_code].get('lastPrice', 0)

        if buy_ref_price <= 0:
            self.logger.error("买入参考价格无效，放弃买入")
            return

        # 计算买入数量
        # 预留 0.2% 资金防止手续费/滑点导致废单
        safe_capital = available_cash * 0.998
        target_volume = int(safe_capital / buy_ref_price / 100) * 100

        if target_volume < 100:
            self.logger.warning(f"计算买入股数不足1手 ({target_volume})，放弃买入")
            return

        self.logger.info(f"执行买入 {target_name}: {target_volume}股")
        buy_seq = self.place_order(target_code, xtconstant.STOCK_BUY, target_volume)

        # 等待买入成交确认
        self.wait_for_order_filled(buy_seq, timeout=45)

        self.logger.info("========== 调仓任务全部结束 ==========")


# ====== 辅助功能函数 ======

def download_daily_data():
    """下载ETF池日线数据"""
    try:
        from miniqmt_etf.etf_momentum_core import ETFMomentumCore
        core = ETFMomentumCore()
        today_str = datetime.now().strftime('%Y%m%d')
        strategy_logger.info(f"=== 开始下载日线数据 ({today_str}) ===")

        for idx, stock_code in enumerate(core.ETF_POOL, 1):
            xtdata.download_history_data(stock_code, '1d', start_time=today_str, end_time=today_str)

        strategy_logger.info("=== 日线数据下载完成 ===")
    except Exception as e:
        strategy_logger.error(f"数据下载异常: {e}")


def monitor_strategy_status():
    """后台监控线程"""
    while True:
        try:
            if xt_trader and acc:
                account_info = xt_trader.query_stock_asset(acc)
                positions = xt_trader.query_stock_positions(acc)
                if account_info:
                    pos_count = len([p for p in positions if p.volume > 0])
                    msg = (f"策略运行中 | 总资产: {account_info.total_asset:.2f} | "
                           f"可用: {account_info.cash:.2f} | 持仓数: {pos_count}")
                    if strategy_logger:
                        strategy_logger.info(msg)
        except Exception:
            pass
        time.sleep(30 * 60)  # 每30分钟打印一次心跳


# ====== 主程序入口 ======

if __name__ == "__main__":
    try:
        # 1. QMT 基础设置
        xtdata.enable_hello = False

        # !!! 请确认此处路径与您的实际环境一致 !!!
        path = r'D:\备份\国金证券QMT交易端\userdata_mini'
        session_id = int(time.time())
        xt_trader = XtQuantTrader(path, session_id)

        # 账号设置
        acc = StockAccount('8886969255', 'STOCK')

        # 2. 启动交易连接
        # 注册回调
        callback = MyXtQuantTraderCallback(query_tool)
        xt_trader.register_callback(callback)

        xt_trader.start()
        res_connect = xt_trader.connect()
        if res_connect != 0:
            print("连接交易端失败，请检查QMT是否登录")
            sys.exit(1)

        res_sub = xt_trader.subscribe(acc)
        if res_sub != 0:
            print("订阅账户失败")
            sys.exit(1)

        # 3. 初始化组件
        strategy_logger = setup_logger()
        strategy_logger.info("===== ETF动量轮动策略启动 (极速实盘版) =====")

        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor_strategy_status, daemon=True)
        monitor_thread.start()

        # 初始化策略实例
        strategy = ETFMomentumStrategy(xt_trader, acc, strategy_logger)

        # 4. 任务调度配置
        scheduler = BackgroundScheduler(timezone="Asia/Shanghai")

        # --- 核心交易任务：11:00:00 准时触发 ---
        scheduler.add_job(
            strategy.execute_rebalance,
            trigger=CronTrigger(hour=11, minute=0, second=0, day_of_week='mon-fri'),
            misfire_grace_time=300,
            id='core_rebalance_task'
        )
        print("任务已装载: 每日 11:00:00 执行 [计算-卖出-等待-买入] 原子化调仓")

        # --- 数据维护任务 ---
        # 16:00 下载策略ETF数据
        scheduler.add_job(
            download_daily_data,
            trigger=CronTrigger(hour=16, minute=0, day_of_week='mon-fri'),
            id='data_dl_etf'
        )
        # 16:05 下载全市场数据(用于其他分析)
        scheduler.add_job(
            data_updater.update_all_daily_data,
            trigger=CronTrigger(hour=16, minute=5, day_of_week='mon-fri'),
            id='data_dl_all'
        )

        # --- 自动退出任务 ---
        def shutdown_and_exit():
            strategy_logger.info("已到收盘时间，程序自动退出。")
            try:
                xt_trader.stop()
                scheduler.shutdown(wait=False)
            except:
                pass
            time.sleep(1)
            os._exit(0)


        # 如果启动时已经过了17点，直接退出
        if datetime.now().hour >= 17:
            print("当前已闭市，程序退出。")
            os._exit(0)

        scheduler.add_job(
            shutdown_and_exit,
            trigger=CronTrigger(hour=17, minute=0, day_of_week='mon-fri'),
            id='auto_shutdown'
        )

        # 5. 启动调度并保持运行
        # 忽略中断信号
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        scheduler.start()

        while True:
            time.sleep(1)

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"主程序崩溃: {err_msg}")
        # 尝试写入错误日志
        try:
            with open('fatal_error.log', 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now()}] {err_msg}\n")
        except:
            pass