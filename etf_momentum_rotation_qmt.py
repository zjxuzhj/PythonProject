# ETF动量轮动策略 - MiniQMT版本
# 基于聚宽策略改写，适用于MiniQMT实盘交易
# 原策略来源：https://www.joinquant.com/post/62008

import threading
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from xtquant import xtconstant, xtdata
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount

from miniqmt_callback import MyXtQuantTraderCallback
from miniqmt_logging_utils import setup_logger
import getAllStockCsv as tools

# ====== 全局配置 ======
TOTAL_BUDGET = 100000  # 总投资预算
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
    "518880.SH",  # 黄金ETF
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
    "159851.SZ",  # 金融科技
    "159637.SZ",  # 新能源车
    "516160.SH",  # 新能源
    "159550.SZ",  # 互联网ETF
    "512710.SH",  # 军工ETF
    "159692.SZ",  # 证券
    "512480.SH",  # 半导体
]

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
            # 转换股票代码格式
            converted_code = self.convert_stock_code(stock_code)

            # 获取历史数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days + 10)).strftime('%Y%m%d')

            data = xtdata.get_market_data_ex(
                stock_list=[converted_code],
                period='1d',
                start_time=start_date,
                end_time=end_date,
                count=-1,
                dividend_type='front',
                fill_data=True
            )

            if converted_code in data and len(data[converted_code]) > 0:
                df = data[converted_code]
                # 确保有足够的数据
                if len(df) >= days:
                    return df.tail(days)

            self.logger.warning(f"获取 {stock_code} 历史数据失败或数据不足")
            return None

        except Exception as e:
            self.logger.error(f"获取 {stock_code} 历史数据异常: {str(e)}")
            return None

    def convert_stock_code(self, stock_code):
        """转换股票代码格式"""
        if '.SH' in stock_code:
            return stock_code.replace('.SH', '.SS')
        elif '.SZ' in stock_code:
            return stock_code.replace('.SZ', '.SZ')
        return stock_code

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
                'current_price': current_price
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

            # 输出所有ETF得分
            self.logger.info("【ETF得分列表（按score降序）】")
            for data in etf_scores:
                try:
                    etf_name = query_tool.get_name_by_code(data['stock_code'])
                    self.logger.info(
                        f"{data['stock_code']} | {etf_name} | "
                        f"得分: {data['score']:.4f} | "
                        f"年化收益: {data['annualized_returns']:.2%} | "
                        f"R²: {data['r2']:.4f} | "
                        f"当前价: {data['current_price']:.2f}"
                    )
                except Exception:
                    self.logger.info(f"{data['stock_code']} | 得分: {data['score']:.4f}")

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
                if pos.m_nVolume > 0:  # 只考虑有持仓的
                    position_dict[pos.stock_code] = {
                        'volume': pos.m_nVolume,
                        'can_use_volume': pos.m_nCanUseVolume,
                        'market_value': pos.m_dMarketValue
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
                return asset.m_dAvailableCash
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

            # 5. 调整目标ETF持仓
            for etf_data in target_etfs:
                stock_code = etf_data['stock_code']
                target_value = total_value * target_weight
                self.execute_trade(stock_code, target_value)

            self.logger.info("=== ETF动量轮动调仓完成 ===")

        except Exception as e:
            self.logger.error(f"调仓异常: {str(e)}")


def etf_rotation_adjust():
    """ETF轮动调仓任务"""
    try:
        strategy = ETFMomentumStrategy(xt_trader, acc, strategy_logger)
        strategy.adjust_positions()
    except Exception as e:
        strategy_logger.error(f"ETF轮动调仓任务异常: {str(e)}")


def monitor_strategy_status():
    """监控策略状态"""
    while True:
        try:
            asset = xt_trader.query_stock_asset(acc)
            positions = xt_trader.query_stock_positions(acc)

            if asset:
                available_cash = asset.m_dAvailableCash
                total_asset = asset.m_dTotalAsset

                position_count = len([p for p in positions if p.m_nVolume > 0])

                status_msg = (
                    f"ETF轮动策略运行正常 | "
                    f"总资产: {total_asset:.2f} | "
                    f"可用资金: {available_cash:.2f} | "
                    f"持仓数量: {position_count}"
                )
                strategy_logger.info(status_msg)

        except Exception as e:
            strategy_logger.error(f"状态监控异常: {str(e)}")

        time.sleep(30 * 60)  # 30分钟监控一次


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

    # 配置定时任务
    scheduler = BackgroundScheduler(timezone="Asia/Shanghai")

    # 获取当前持仓股票集合
    positions = xt_trader.query_stock_positions(acc)
    hold_stocks = {pos.stock_code for pos in positions}

    # 每日11:00执行调仓
    scheduler.add_job(
        etf_rotation_adjust,
        trigger=CronTrigger(
            hour=11,
            minute=0,
            day_of_week='mon-fri'
        ),
        misfire_grace_time=300
    )
    print("定时任务已启动：每日11:00执行ETF动量轮动调仓")

    # 启动调度器
    scheduler.start()

    # 保持程序运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序退出")
        scheduler.shutdown()