# ETF动量轮动策略 - MiniQMT版本
# 基于聚宽策略改写，适用于MiniQMT实盘交易
# 原策略来源：https://www.joinquant.com/post/62008
# 现在使用统一的ETF核心模块

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

import getAllStockCsv as tools
import updateAllStockDataCache as data_updater
from miniqmt_etf.etf_momentum_core import ETFMomentumCore
from miniqmt_etf.miniqmt_callback import MyXtQuantTraderCallback
from miniqmt_etf.miniqmt_logging_utils import setup_logger
import os

# ====== 全局配置 ======
_env_alloc = os.environ.get('ALLOCATED_CAPITAL')
TOTAL_BUDGET = float(_env_alloc) if _env_alloc else 5000  # 总投资预算（可由主控程序覆盖）
MIN_TRADE_AMOUNT = 1000  # 最小交易金额
MOMENTUM_DAYS = 25  # 动量计算天数
MAX_ETF_COUNT = 1  # 最大持有ETF数量

# ETF池配置和名称映射现在由统一的核心模块提供

# 全局变量
query_tool = tools.StockQuery()
strategy_logger = None
xt_trader = None
acc = None
available_cash = 0
current_positions = {}


class ETFMomentumStrategy:
    """ETF动量轮动策略类"""

    def __init__(self, trader, account, logger):
        self.trader = trader
        self.account = account
        self.logger = logger
        self.core = ETFMomentumCore(MOMENTUM_DAYS)  # 使用统一的核心模块
        # 支持环境变量覆盖最大ETF数量（来自主控程序）
        try:
            env_max = os.environ.get('MAX_POSITIONS')
            self.max_etf_count = int(env_max) if env_max else MAX_ETF_COUNT
        except Exception:
            self.max_etf_count = MAX_ETF_COUNT

    def filter_etfs(self):
        """筛选ETF并计算得分"""
        try:
            self.logger.info("开始计算ETF动量得分...")

            # 使用核心模块获取ETF得分
            etf_scores = self.core.get_etf_scores_for_live()

            # 打印详细表格
            self.core.print_etf_scores(etf_scores)

            # 保留日志标题，但注释掉原有的逐行日志记录，避免信息重复
            self.logger.info("【ETF得分列表（按score降序）】 - 详细见上方控制台表格")

            # 过滤有效得分的ETF
            valid_etfs = [data for data in etf_scores if 0 < data['score'] < 6]

            return valid_etfs[:self.max_etf_count]

        except Exception as e:
            self.logger.error(f"筛选ETF异常: {str(e)}")
            return []

    def get_current_positions(self):
        """获取当前持仓"""
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
        """获取可用资金"""
        try:
            asset = self.trader.query_stock_asset(self.account)
            if asset:
                return asset.cash
            return 0
        except Exception as e:
            self.logger.error(f"获取可用资金异常: {str(e)}")
            return 0

    def execute_trade(self, stock_code, target_value):
        """执行交易"""
        try:
            tick_data = xtdata.get_full_tick([stock_code])
            if not tick_data or stock_code not in tick_data or not tick_data[stock_code]:
                self.logger.warning(f"{stock_code} 获取tick五档行情失败，跳过交易")
                return False
            tick = tick_data[stock_code]

            # 2. 检查五档行情列表是否存在且长度足够
            bid_prices = tick.get('bidPrice', [])
            ask_prices = tick.get('askPrice', [])
            if len(bid_prices) < 5 or len(ask_prices) < 5:
                self.logger.warning(f"{stock_code} 五档行情数据不完整，跳过交易")
                return False

            # 3. 使用最新成交价作为计算目标股数的“参考价”
            reference_price = tick.get('lastPrice', 0)
            if reference_price <= 0:
                self.logger.warning(f"{stock_code} 参考价格无效，跳过交易")
                return False

            # 4. 计算交易量
            current_positions = self.get_current_positions()
            current_volume = current_positions.get(stock_code, {}).get('volume', 0)
            target_volume = int(target_value / reference_price / 100) * 100
            volume_diff = target_volume - current_volume

            if abs(volume_diff * reference_price) < MIN_TRADE_AMOUNT:
                self.logger.info(f"{stock_code} 调整金额过小，跳过交易")
                return True

            # 执行交易
            if volume_diff > 0:  # 买入
                execution_price = ask_prices[4] or reference_price
                if execution_price <= 0:
                    self.logger.warning(f"{stock_code} 获取买入执行价（卖五）失败，跳过交易")
                    return False

                available_cash = self.get_available_cash()
                required_cash = volume_diff * execution_price  # 使用执行价计算实际成本
                if required_cash > available_cash:
                    self.logger.info(
                        f"按计划买入({volume_diff}股)所需资金({required_cash:.2f}) > 可用现金({available_cash:.2f})，重新计算股数...")
                    volume_diff = int(available_cash / execution_price / 100) * 100  # 根据可用现金和执行价计算能买的最大股数
                    # 重新计算后再次检查
                    if volume_diff <= 0:
                        self.logger.warning(f"{stock_code} 资金不足，无法买入")
                        return False
                    # 检查调整后的金额是否仍满足最小交易额
                    if (volume_diff * execution_price) < MIN_TRADE_AMOUNT:
                        self.logger.warning(f"{stock_code} 资金不足以满足最小交易金额，取消买入")
                        return False
                # 执行买入
                self.trader.order_stock_async(
                    self.account, stock_code, xtconstant.STOCK_BUY,
                    volume_diff, xtconstant.FIX_PRICE, execution_price,
                    'ETF动量轮动买入', ''
                )
                etf_name = self.core.etf_names.get(stock_code, '未知')
                self.logger.info(f"买入: {etf_name}({stock_code}) {volume_diff}股 @ {execution_price:.3f} (卖五价)")

            elif volume_diff < 0:  # 卖出
                # 从 bidPrice 列表中获取买五价
                execution_price = bid_prices[4] or reference_price
                if execution_price <= 0:
                    self.logger.warning(f"{stock_code} 获取卖出执行价（买五）失败，跳过交易")
                    return False

                sell_volume = abs(volume_diff)
                can_use_volume = current_positions.get(stock_code, {}).get('can_use_volume', 0)
                if sell_volume > can_use_volume:
                    sell_volume = can_use_volume

                if sell_volume > 0:
                    # 执行卖出
                    self.trader.order_stock_async(
                        self.account, stock_code, xtconstant.STOCK_SELL,
                        sell_volume, xtconstant.FIX_PRICE, execution_price,
                        'ETF动量轮动卖出', ''
                    )
                    etf_name = self.core.etf_names.get(stock_code, '未知')
                    self.logger.info(f"卖出: {etf_name}({stock_code}) {sell_volume}股 @ {execution_price:.3f} (买五价)")

            return True

        except Exception as e:
            self.logger.error(f"执行 {stock_code} 交易异常: {str(e)}")
            return False

    def handle_sells(self):
        """处理卖出逻辑"""
        self.logger.info("=== [1/2] 开始执行卖出检查 ===")
        target_etfs = self.filter_etfs()  # 先计算出目标ETF
        if not target_etfs:
            self.logger.warning("未找到目标ETF，不执行卖出。")
            return

        target_codes = {etf['stock_code'] for etf in target_etfs}
        current_positions = self.get_current_positions()

        # 遍历当前持仓，如果不在目标里，就清仓
        for stock_code in current_positions:
            # 该持仓在策略ETF池中，它不是今天的目标ETF（也就是需要卖出的票）
            if stock_code in self.core.ETF_POOL:
                stock_name = self.core.etf_names.get(stock_code, '未知')
                if stock_code in target_codes:
                    self.logger.info(f"【持仓检查】当前持仓 '{stock_name}({stock_code})' 与目标一致，无需卖出。")
                else:
                    self.logger.info(f"【卖出决策】标的 '{stock_name}({stock_code})' 已非目标，准备清仓。")
                    self.execute_trade(stock_code, 0)  # 目标市值传0，即清仓

    def handle_buys(self):
        """处理买入逻辑"""
        self.logger.info("=== [2/2] 开始执行买入检查 ===")

        # 1. 再次计算，确保逻辑一致性
        target_etfs = self.filter_etfs()
        if not target_etfs:
            self.logger.warning("未找到目标ETF，不执行买入。")
            return

        target_code = target_etfs[0]['stock_code']
        etf_name = self.core.etf_names.get(target_code, '未知')

        # 2. 获取当前持仓并检查是否已持有目标
        current_positions = self.get_current_positions()
        if target_code in current_positions:
            # 【关键优化】打印更清晰的提示日志
            self.logger.info(f"【持仓检查】已持有目标ETF '{etf_name}({target_code})'，无需执行买入操作。")
            return

        # 3. 获取当前持仓和可用资金，计算账户总资产，用于确定买入金额
        available_cash = self.get_available_cash()
        total_value = available_cash
        for pos_data in current_positions.values():
            total_value += pos_data['market_value']

        self.logger.info(f"账户总资产: {total_value:.2f}, 可用现金: {available_cash:.2f}")
        self.logger.info(f"【买入决策】准备买入 '{etf_name}({target_code})'")

        # 买入目标ETF
        self.execute_trade(target_code, total_value)

        self.logger.info("=== ETF动量轮动调仓完成 ===")


def download_daily_data():
    """下载ETF池中所有证券的最新日线数据"""
    try:
        from miniqmt_etf.etf_momentum_core import ETFMomentumCore
        core = ETFMomentumCore()

        today_str = datetime.now().strftime('%Y%m%d')
        strategy_logger.info(f"=== 开始执行每日日线数据下载任务 ({today_str}) ===")

        total_etfs = len(core.ETF_POOL)
        for idx, stock_code in enumerate(core.ETF_POOL, 1):
            strategy_logger.info(
                f"[{idx}/{total_etfs}] 下载 {stock_code} ({core.etf_names.get(stock_code, '未知')}) 的日线数据...")
            # 下载当天的日线数据，会自动更新到本地数据文件
            xtdata.download_history_data(stock_code, '1d', start_time=today_str, end_time=today_str)

        strategy_logger.info("=== 所有ETF日线数据下载完成 ===")

    except Exception as e:
        strategy_logger.error(f"每日数据下载任务异常: {str(e)}")


def monitor_strategy_status():
    """监控策略状态"""
    while True:
        try:
            account_info = xt_trader.query_stock_asset(acc)
            positions = xt_trader.query_stock_positions(acc)

            if account_info:
                available_cash = account_info.cash
                total_asset = account_info.total_asset

                position_count = len([p for p in positions if p.volume > 0])  # 同样修改 p.m_nVolume

                status_msg = (
                    f"ETF轮动策略运行正常 | "
                    f"总资产: {total_asset:.2f} | "
                    f"可用资金: {available_cash:.2f} | "
                    f"持仓数量: {position_count}"
                )
                strategy_logger.info(status_msg)

        except Exception as e:
            strategy_logger.error(f"状态监控异常: {str(e)}")

        time.sleep(30 * 60)


def verify_data_download(check_date_str, stock_list):
    """
    验证指定日期的本地数据是否下载成功
    :param check_date_str: 要检查的日期字符串, 格式 'YYYYMMDD'
    :param stock_list: 要检查的股票代码列表
    """
    try:
        strategy_logger.info(f"--- 开始验证 {check_date_str} 的本地数据 ---")

        # 从列表中随机选择几只ETF进行抽查，这里我们只查前3只
        sample_etfs = stock_list[:3]

        for stock_code in sample_etfs:
            data = xtdata.get_local_data(
                field_list=['open', 'high', 'low', 'close'],  # 读取OHLC数据
                stock_list=[stock_code],
                period='1d',
                start_time=check_date_str,
                end_time=check_date_str
            )

            if stock_code in data and not data[stock_code].empty:
                # 如果能读取到数据，说明下载成功
                latest_data = data[stock_code].iloc[0]  # 获取第一行数据
                strategy_logger.info(f" 验证成功: {stock_code} 在 {check_date_str} 的数据已存在。")
                print(f" {stock_code} 数据抽查: Close={latest_data['close']}")  # 打印收盘价以供核对
            else:
                strategy_logger.warning(f" 验证失败: 未找到 {stock_code} 在 {check_date_str} 的本地数据。")
                print(f" {stock_code} 数据抽查失败。")

        strategy_logger.info("--- 数据验证完成 ---")

    except Exception as e:
        strategy_logger.error(f"数据验证过程中发生异常: {str(e)}")


if __name__ == "__main__":
    # 初始化
    xtdata.enable_hello = False
    path = r'D:\备份\国金证券QMT交易端\userdata_mini'
    session_id = int(time.time())
    xt_trader = XtQuantTrader(path, session_id)

    # 账户配置
    acc = StockAccount('8886969255', 'STOCK')

    # 注册回调
    callback = MyXtQuantTraderCallback(query_tool)
    xt_trader.register_callback(callback)

    # 启动交易
    xt_trader.start()

    # 建立连接
    connect_result = xt_trader.connect()
    print('建立交易连接，返回0表示连接成功', connect_result)

    # 订阅交易回调
    subscribe_result = xt_trader.subscribe(acc)
    print('对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功', subscribe_result)

    # 初始化日志
    strategy_logger = setup_logger()
    strategy_logger.info("===== ETF动量轮动策略启动 =====")

    # 启动状态监控线程
    monitor_thread = threading.Thread(
        target=monitor_strategy_status,
        daemon=True
    )
    monitor_thread.start()

    # 实例化策略对象
    strategy = ETFMomentumStrategy(xt_trader, acc, strategy_logger)

    # 配置定时任务
    scheduler = BackgroundScheduler(timezone="Asia/Shanghai")

    # download_daily_data()
    # yesterday = datetime.now() - timedelta(days=1)
    # today_str_for_verify = datetime.now().strftime('%Y%m%d')
    # core = ETFMomentumCore()
    # verify_data_download(check_date_str=today_str_for_verify, stock_list=core.ETF_POOL)

    # 任务1：每日10:58执行卖出
    scheduler.add_job(
        strategy.handle_sells,  # 调用卖出函数
        trigger=CronTrigger(
            hour=10,
            minute=58,
            day_of_week='mon-fri'
        ),
        misfire_grace_time=300,
        id='etf_sells_handler'
    )
    print("定时任务已启动：每日10:58执行ETF卖出检查")

    # 任务2：每日11:00执行买入
    scheduler.add_job(
        strategy.handle_buys,  # 调用买入函数
        trigger=CronTrigger(
            hour=11,
            minute=0,
            day_of_week='mon-fri'
        ),
        misfire_grace_time=300,
        id='etf_buys_handler'
    )
    print("定时任务已启动：每日11:00执行ETF买入检查")

    # 每日16:00执行数据下载任务
    scheduler.add_job(
        download_daily_data,
        trigger=CronTrigger(
            hour=16,
            minute=0,
            day_of_week='mon-fri'  # 周一至周五执行
        ),
        id='daily_data_downloader',  # 给任务一个唯一的ID
        misfire_grace_time=300
    )
    print("定时任务已启动：每日16:00执行ETF动量轮动日线数据下载")

    scheduler.add_job(
        data_updater.update_all_daily_data,
        trigger=CronTrigger(
            hour=16,
            minute=5,
            day_of_week='mon-fri'
        ),
        id='daily_data_small_downloader',
        misfire_grace_time=300
    )
    print("定时任务已启动：每日16:05执行小市值策略日线数据下载")

    # 忽略父进程的 Ctrl-C 信号，避免非预期退出
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    except Exception:
        pass

    # 启动调度器
    scheduler.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        try:
            strategy_logger and strategy_logger.info("忽略Ctrl-C信号，继续运行")
        except Exception:
            pass
        while True:
            time.sleep(1)
    except Exception:
        err = traceback.format_exc()
        sys.stderr.write(err + "\n")
        try:
            os.makedirs('analysis_results', exist_ok=True)
            with open(os.path.join('analysis_results', 'etf_runner_error.log'), 'a', encoding='utf-8') as f:
                f.write(err + "\n")
        except Exception:
            pass
