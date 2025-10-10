# ETF动量轮动策略 - MiniQMT版本
# 基于聚宽策略改写，适用于MiniQMT实盘交易
# 原策略来源：https://www.joinquant.com/post/62008

import math
import threading
import time
from datetime import datetime, timedelta

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from xtquant import xtconstant, xtdata
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount

import getAllStockCsv as tools
from miniqmt_callback import MyXtQuantTraderCallback
from miniqmt_logging_utils import setup_logger
import updateAllStockDataCache as data_updater

# ====== 全局配置 ======
TOTAL_BUDGET = 5000  # 总投资预算
MIN_TRADE_AMOUNT = 1000  # 最小交易金额
MOMENTUM_DAYS = 25  # 动量计算天数
MAX_ETF_COUNT = 1  # 最大持有ETF数量

# ETF池配置
ETF_POOL = [
    # 境外
    "513100.SH",  # 纳指ETF
    "159509.SZ",  # 纳指科技ETF
    "513520.SH",  # 日经ETF
    "513030.SH",  # 德国ETF
    # 商品
    "518880.SH",  # 黄金ETF 2013.7 714亿
    "159980.SZ",  # 有色ETF
    "159985.SZ",  # 豆粕ETF
    "501018.SH",  # 南方原油
    # 债券
    "511090.SH",  # 30年国债ETF
    # 国内
    "513130.SH",  # 恒生科技
    "513690.SH",  # 港股红利
    "510180.SH",  # 上证180
    "159915.SZ",  # 创业板ETF
    "510410.SH",  # 资源
    "515650.SH",  # 消费50
    "512290.SH",  # 生物医药
    "588120.SH",  # 科创100
    "515070.SH",  # 人工智能ETF
    "159851.SZ",  # 金融科技 2021.3 123亿
    "159637.SZ",  # 新能源车
    "516160.SH",  # 新能源 2021.2 61亿 中证新能
    "159550.SZ",  # 互联网ETF
    "512710.SH",  # 军工ETF
    "159692.SZ",  # 证券
    "512480.SH",  # 半导体 2019.6 230亿
]

ETF_NAMES = {
    "513100.SH": "纳指ETF",
    "159509.SZ": "纳指科技ETF",
    "513520.SH": "日经ETF",
    "513030.SH": "德国ETF",
    "518880.SH": "黄金ETF",
    "159980.SZ": "有色ETF",
    "159985.SZ": "豆粕ETF",
    "501018.SH": "南方原油",
    "511090.SH": "30年国债ETF",
    "513130.SH": "恒生科技",
    "513690.SH": "港股红利",
    "510180.SH": "上证180",
    "159915.SZ": "创业板ETF",
    "510410.SH": "资源",
    "515650.SH": "消费50",
    "512290.SH": "生物医药",
    "588120.SH": "科创100",
    "515070.SH": "人工智能ETF",
    "159851.SZ": "金融科技",
    "159637.SZ": "新能源车",
    "516160.SH": "新能源",
    "159550.SZ": "互联网ETF",
    "512710.SH": "军工ETF",
    "159692.SZ": "证券",
    "512480.SH": "半导体",
}

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
        self.etf_pool = ETF_POOL
        self.momentum_days = MOMENTUM_DAYS
        self.max_etf_count = MAX_ETF_COUNT

    def get_historical_data(self, stock_code, days):
        """获取历史数据"""
        try:
            # 获取历史数据
            yesterday = datetime.now() - timedelta(days=1)
            end_date = yesterday.strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days + 40)).strftime('%Y%m%d')
            data = xtdata.get_local_data(
                field_list=['close'],
                stock_list=[stock_code],
                period='1d',
                start_time=start_date,
                end_time=end_date,
            )

            if stock_code in data and len(data[stock_code]) > 0:
                df = data[stock_code]
                # 确保有足够的数据
                if len(df) >= days:
                    return df.tail(days)

            self.logger.warning(f"获取 {stock_code} 历史数据失败或数据不足")
            return None

        except Exception as e:
            self.logger.error(f"获取 {stock_code} 历史数据异常: {str(e)}")
            return None

    def get_current_price(self, stock_code):
        """获取当前价格"""
        try:
            tick = xtdata.get_full_tick([stock_code])
            if stock_code in tick and tick[stock_code]:
                return tick[stock_code].get('lastPrice', 0)
            return 0
        except Exception as e:
            self.logger.error(f"获取 {stock_code} 当前价格异常: {str(e)}")
            return 0

    def calculate_momentum_score(self, stock_code):
        """计算动量得分"""
        try:
            # 获取历史数据
            df = self.get_historical_data(stock_code, self.momentum_days)
            if df is None or len(df) < self.momentum_days:
                return None

            # 获取当前价格
            current_price = self.get_current_price(stock_code)
            if current_price <= 0:
                return None

            # 构建价格序列（历史价格 + 当前价格）
            prices = np.append(df['close'].values, current_price)

            # 计算对数价格
            y = np.log(prices)
            x = np.arange(len(y))

            # 设置权重（线性递增）
            weights = np.linspace(1, 2, len(y))

            # 加权线性回归
            slope, intercept = np.polyfit(x, y, 1, w=weights)

            # 计算年化收益率
            annualized_returns = math.exp(slope * 250) - 1

            # 计算R²
            y_pred = slope * x + intercept
            ss_res = np.sum(weights * (y - y_pred) ** 2)
            ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # 计算得分
            score = annualized_returns * r2

            # 过滤近3日跌幅超过5%的ETF
            if len(prices) >= 4:
                recent_drops = [
                    prices[-1] / prices[-2],
                    prices[-2] / prices[-3],
                    prices[-3] / prices[-4]
                ]
                if min(recent_drops) < 0.95:
                    score = 0

            return {
                'stock_code': stock_code,
                'annualized_returns': annualized_returns,
                'r2': r2,
                'score': score,
                'current_price': current_price,
                'data_points': len(df)
            }

        except Exception as e:
            self.logger.error(f"计算 {stock_code} 动量得分异常: {str(e)}")
            return None

    def filter_etfs(self):
        """筛选ETF并计算得分"""
        try:
            self.logger.info("开始计算ETF动量得分...")

            etf_scores = []

            for etf_code in self.etf_pool:
                score_data = self.calculate_momentum_score(etf_code)
                if score_data:
                    etf_scores.append(score_data)

            # 按得分排序
            etf_scores.sort(key=lambda x: x['score'], reverse=True)

            # 获取当前日期用于显示
            today_str = datetime.now().strftime('%Y-%m-%d')
            print(f"\n{'='*75}")
            print(f"ETF分值排序结果 ({today_str})")
            print(f"{'='*75}")
            print(f"{'排名':<4} {'代码':<12} {'名称':<12} {'分值':<8} {'年化收益':<10} {'R2':<8} {'价格':<8} {'数据点'}")
            print("-" * 75)

            for rank, data in enumerate(etf_scores, 1):
                etf_name = ETF_NAMES.get(data['stock_code'], '未知')
                print(f"{rank:<4} {data['stock_code']:<12} {etf_name:<12} "
                      f"{data['score']:<8.4f} {data['annualized_returns']:<10.2%} "
                      f"{data['r2']:<8.4f} {data['current_price']:<8.2f} {data['data_points']}")
            print(f"{'='*75}\n")


            # 保留日志标题，但注释掉原有的逐行日志记录，避免信息重复
            self.logger.info("【ETF得分列表（按score降序）】 - 详细见上方控制台表格")
            # for data in etf_scores:
            #     try:
            #         etf_name = ETF_NAMES.get(data['stock_code'], '未知')
            #         self.logger.info(
            #             f"{data['stock_code']} | {etf_name} | "
            #             f"得分: {data['score']:.4f} | "
            #             f"年化收益: {data['annualized_returns']:.2%} | "
            #             f"R2: {data['r2']:.4f} | "
            #             f"当前价: {data['current_price']:.2f}"
            #         )
            #     except Exception:
            #         self.logger.info(f"{data['stock_code']} | 得分: {data['score']:.4f}")

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
            current_price = self.get_current_price(stock_code)
            if current_price <= 0:
                self.logger.warning(f"{stock_code} 价格无效，跳过交易")
                return False

            # 获取当前持仓
            current_positions = self.get_current_positions()
            current_value = 0
            current_volume = 0

            if stock_code in current_positions:
                current_volume = current_positions[stock_code]['volume']
                current_value = current_volume * current_price

            # 计算目标持仓数量
            target_volume = int(target_value / current_price / 100) * 100

            # 计算需要调整的数量
            volume_diff = target_volume - current_volume

            if abs(volume_diff * current_price) < MIN_TRADE_AMOUNT:
                self.logger.info(f"{stock_code} 调整金额过小，跳过交易")
                return True

            # 执行交易
            if volume_diff > 0:  # 买入
                available_cash = self.get_available_cash()
                trade_value = volume_diff * current_price

                if trade_value > available_cash:
                    volume_diff = int(available_cash / current_price / 100) * 100
                    if volume_diff <= 0:
                        self.logger.warning(f"{stock_code} 资金不足，无法买入")
                        return False

                # 执行买入
                order_id = self.trader.order_stock_async(
                    self.account,
                    stock_code,
                    xtconstant.STOCK_BUY,
                    volume_diff,
                    xtconstant.FIX_PRICE,
                    current_price,
                    'ETF动量轮动买入',
                    ''
                )

                etf_name = query_tool.get_name_by_code(stock_code)
                self.logger.info(f"买入: {etf_name}({stock_code}) {volume_diff}股 @ {current_price:.2f}")

            elif volume_diff < 0:  # 卖出
                sell_volume = abs(volume_diff)
                can_use_volume = current_positions[stock_code]['can_use_volume']

                if sell_volume > can_use_volume:
                    sell_volume = can_use_volume

                if sell_volume > 0:
                    # 执行卖出
                    order_id = self.trader.order_stock_async(
                        self.account,
                        stock_code,
                        xtconstant.STOCK_SELL,
                        sell_volume,
                        xtconstant.FIX_PRICE,
                        current_price,
                        'ETF动量轮动卖出',
                        ''
                    )

                    etf_name = query_tool.get_name_by_code(stock_code)
                    self.logger.info(f"卖出: {etf_name}({stock_code}) {sell_volume}股 @ {current_price:.2f}")

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
            # 必须同时满足1.该持仓在我们的策略ETF池中 2.它不是今天的目标ETF
            if stock_code in self.etf_pool and stock_code not in target_codes:
                self.logger.info(f"持仓 {stock_code} 在策略池中但非今日目标，执行卖出。")
                self.execute_trade(stock_code, 0)  # 目标市值传0，即清仓

    def handle_buys(self):
        """处理买入逻辑"""
        self.logger.info("=== [2/2] 开始执行买入检查 ===")
        target_etfs = self.filter_etfs()  # 再次计算，确保逻辑一致
        if not target_etfs:
            self.logger.warning("未找到目标ETF，不执行买入。")
            return

        # 2. 获取当前持仓和可用资金
        current_positions = self.get_current_positions()
        available_cash = self.get_available_cash()

        # 计算总可用资金（持仓市值 + 可用现金）
        total_value = available_cash
        for pos_data in current_positions.values():
            total_value += pos_data['market_value']

        self.logger.info(f"总资产: {total_value:.2f}, 可用现金: {available_cash:.2f}")

        # 3. 计算目标配置
        target_codes = [etf['stock_code'] for etf in target_etfs]
        target_weight = 1.0 / len(target_codes)

        self.logger.info(f"目标ETF: {target_codes}, 每个权重: {target_weight:.2%}")

        # 买入目标ETF
        for etf_data in target_etfs:
            stock_code = etf_data['stock_code']
            target_value = total_value * target_weight
            self.execute_trade(stock_code, target_value)

        self.logger.info("=== ETF动量轮动调仓完成 ===")

    def adjust_positions(self):
        """调仓主函数"""
        try:
            self.logger.info("=== 开始ETF动量轮动调仓 ===")

            # 1. 筛选目标ETF
            target_etfs = self.filter_etfs()

            if not target_etfs:
                self.logger.warning("未找到符合条件的ETF，跳过调仓")
                return

            # 2. 获取当前持仓和可用资金
            current_positions = self.get_current_positions()
            available_cash = self.get_available_cash()

            # 计算总可用资金（持仓市值 + 可用现金）
            total_value = available_cash
            for pos_data in current_positions.values():
                total_value += pos_data['market_value']

            self.logger.info(f"总资产: {total_value:.2f}, 可用现金: {available_cash:.2f}")

            # 3. 计算目标配置
            target_codes = [etf['stock_code'] for etf in target_etfs]
            target_weight = 1.0 / len(target_codes)

            self.logger.info(f"目标ETF: {target_codes}, 每个权重: {target_weight:.2%}")

            # 4. 卖出不在目标列表中的持仓 暂时先不清仓
            # for stock_code in current_positions:
            #     if stock_code not in target_codes:
            #         self.execute_trade(stock_code, 0)  # 清仓

            # 5. 调整目标ETF持仓 暂时先不买入卖出
            for etf_data in target_etfs:
                stock_code = etf_data['stock_code']
                target_value = total_value * target_weight
                self.execute_trade(stock_code, target_value)

            self.logger.info("=== ETF动量轮动调仓完成 ===")

        except Exception as e:
            self.logger.error(f"调仓异常: {str(e)}")


def download_daily_data():
    """下载ETF池中所有证券的最新日线数据"""
    try:
        today_str = datetime.now().strftime('%Y%m%d')
        strategy_logger.info(f"=== 开始执行每日日线数据下载任务 ({today_str}) ===")

        total_etfs = len(ETF_POOL)
        for idx, stock_code in enumerate(ETF_POOL, 1):
            strategy_logger.info(
                f"[{idx}/{total_etfs}] 下载 {stock_code} ({ETF_NAMES.get(stock_code, '未知')}) 的日线数据...")
            # 下载当天的日线数据，会自动更新到本地数据文件
            xtdata.download_history_data(stock_code, '1d', start_time=today_str, end_time=today_str)

        strategy_logger.info("=== 所有ETF日线数据下载完成 ===")

    except Exception as e:
        strategy_logger.error(f"每日数据下载任务异常: {str(e)}")


def monitor_strategy_status():
    """监控策略状态"""
    while True:
        try:
            asset = xt_trader.query_stock_asset(acc)
            positions = xt_trader.query_stock_positions(acc)

            if asset:
                available_cash = asset.cash  # 修改此处
                total_asset = asset.total_asset  # 修改此处

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
        data_updater.update_all_daily_data(),
        trigger=CronTrigger(
            hour=16,
            minute=5,
            day_of_week='mon-fri'
        ),
        id='daily_data_small_downloader',
        misfire_grace_time=300
    )
    print("定时任务已启动：每日16:00执行小市值策略日线数据下载")

    # download_daily_data()
    # today_str_for_verify = datetime.now().strftime('%Y%m%d')
    # verify_data_download(check_date_str=today_str_for_verify, stock_list=ETF_POOL)

    # 任务1：每日10:55执行卖出
    scheduler.add_job(
        strategy.handle_sells,  # 调用卖出函数
        trigger=CronTrigger(
            hour=10,
            minute=55,
            day_of_week='mon-fri'
        ),
        misfire_grace_time=300,
        id='etf_sells_handler'
    )
    print("定时任务已启动：每日10:55执行ETF卖出检查")

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

    # 启动调度器
    scheduler.start()

    # 保持程序运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序退出")
        scheduler.shutdown()
