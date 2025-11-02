# -*- coding: utf-8 -*-
"""
聚宽首板回踩MA5策略
基于miniqmt实盘策略改编的聚宽回测策略

策略逻辑：
1. 股票池选择：筛选当日首板涨停股票，建立5日跟踪池
2. 买入时机：首板后第2天或第4天，价格回踩至MA5附近（±2%）
3. 卖出时机：收盘价跌破MA5、持仓超过15天、炸板等条件

作者：基于miniqmt策略改编
日期：2024年
"""

# 导入聚宽函数库
import jqdata
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入自定义模块
from joinquant_stock_pool import StockPoolManager
from joinquant_buy_timing import BuyTimingManager  
from joinquant_sell_timing import SellTimingManager


# 全局变量
g = {}


def initialize(context):
    """
    初始化函数，设定基准等等
    """
    log.info("=== 首板回踩MA5策略初始化 ===")
    
    # 设定沪深300作为基准
    g.benchmark = '000300.XSHG'
    
    # 设定基准
    set_benchmark(g.benchmark)
    
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')
    
    # 初始化策略组件
    g.stock_pool_manager = StockPoolManager()
    g.buy_timing_manager = BuyTimingManager()
    g.sell_timing_manager = SellTimingManager()
    
    # 策略配置
    g.max_positions = 10  # 最大持仓数量
    g.position_ratio = 0.10  # 单只股票仓位比例
    
    # 统计变量
    g.trade_count = 0
    g.win_count = 0
    g.total_return = 0.0
    
    # 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，无需具体对应）
    run_daily(trade_logic, time='09:30', reference_security='000300.XSHG')
    run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')
    
    log.info("策略初始化完成")


def trade_logic(context):
    """
    主要交易逻辑
    """
    current_date = context.current_dt.date()
    log.info(f"=== {current_date} 交易逻辑开始 ===")
    
    try:
        # 1. 更新股票池（筛选首板股票并更新跟踪池）
        target_stocks = g.stock_pool_manager.get_target_stocks(context, current_date)
        log.info(f"当前跟踪池股票数量: {len(target_stocks)}")
        
        # 2. 执行卖出逻辑
        execute_sell_logic(context, current_date)
        
        # 3. 执行买入逻辑
        execute_buy_logic(context, current_date)
        
        # 4. 记录当日状态
        log_daily_status(context, current_date)
        
    except Exception as e:
        log.error(f"交易逻辑执行出错: {e}")


def execute_sell_logic(context, current_date):
    """
    执行卖出逻辑
    """
    log.info("--- 执行卖出逻辑 ---")
    
    # 获取卖出候选股票
    sell_candidates = g.sell_timing_manager.get_sell_candidates(context, current_date)
    
    if not sell_candidates:
        log.info("无需卖出的股票")
        return
    
    # 执行卖出
    for candidate in sell_candidates:
        stock_code = candidate['code']
        reason = candidate['reason']
        
        # 执行卖出订单
        success = g.sell_timing_manager.execute_sell_order(context, stock_code, reason)
        
        if success:
            # 更新统计
            g.trade_count += 1
            
            # 计算收益（简化处理）
            position = context.portfolio.positions[stock_code]
            if position.avg_cost > 0:
                current_price = get_current_data()[stock_code].last_price
                return_rate = (current_price - position.avg_cost) / position.avg_cost
                g.total_return += return_rate
                
                if return_rate > 0:
                    g.win_count += 1
                
                log.info(f"卖出成功: {stock_code}, 原因: {reason}, 收益率: {return_rate*100:.2f}%")


def execute_buy_logic(context, current_date):
    """
    执行买入逻辑
    """
    log.info("--- 执行买入逻辑 ---")
    
    # 检查是否还能买入（仓位限制）
    current_positions = len([pos for pos in context.portfolio.positions.values() if pos.total_amount > 0])
    
    if current_positions >= g.max_positions:
        log.info(f"已达最大持仓数量限制: {current_positions}/{g.max_positions}")
        return
    
    # 获取买入候选股票
    buy_candidates = g.buy_timing_manager.get_buy_candidates(
        context, g.stock_pool_manager, current_date
    )
    
    if not buy_candidates:
        log.info("无符合买入条件的股票")
        return
    
    log.info(f"发现 {len(buy_candidates)} 只符合买入条件的股票")
    
    # 按优先级买入（优先第2天的股票）
    bought_count = 0
    max_buy_count = g.max_positions - current_positions
    
    for candidate in buy_candidates[:max_buy_count]:
        stock_code = candidate['code']
        stock_name = candidate['name']
        day_count = candidate['day_count']
        reason = candidate['reason']
        
        # 执行买入订单
        success = g.buy_timing_manager.execute_buy_order(context, stock_code, current_date)
        
        if success:
            bought_count += 1
            log.info(f"买入成功: {stock_code} {stock_name}, {reason}")
        else:
            log.warning(f"买入失败: {stock_code} {stock_name}")
    
    log.info(f"本日买入股票数量: {bought_count}")


def log_daily_status(context, current_date):
    """
    记录每日状态
    """
    # 统计当前持仓
    positions = [pos for pos in context.portfolio.positions.values() if pos.total_amount > 0]
    total_positions = len(positions)
    
    # 统计跟踪池
    tracking_count = len(g.stock_pool_manager.tracking_pool)
    
    # 计算当日收益率
    daily_return = (context.portfolio.total_value / context.portfolio.starting_cash - 1) * 100
    
    log.info(f"=== {current_date} 每日状态 ===")
    log.info(f"当前持仓: {total_positions}只")
    log.info(f"跟踪池: {tracking_count}只")
    log.info(f"总资产: {context.portfolio.total_value:.2f}")
    log.info(f"累计收益率: {daily_return:.2f}%")
    
    # 详细持仓信息
    if positions:
        log.info("--- 当前持仓详情 ---")
        for stock_code, position in context.portfolio.positions.items():
            if position.total_amount > 0:
                current_price = get_current_data()[stock_code].last_price
                return_rate = (current_price - position.avg_cost) / position.avg_cost * 100
                
                # 获取股票在跟踪池中的信息
                day_count = g.stock_pool_manager.get_stock_day_count(stock_code)
                
                log.info(f"  {stock_code}: 持仓{position.total_amount}, 成本{position.avg_cost:.2f}, "
                        f"现价{current_price:.2f}, 收益{return_rate:.2f}%, 第{day_count}天")


def after_market_close(context):
    """
    收盘后运行函数
    """
    current_date = context.current_dt.date()
    
    # 清理跟踪池中的过期股票
    g.stock_pool_manager._clean_expired_stocks()
    
    # 输出策略统计信息
    if g.trade_count > 0:
        win_rate = g.win_count / g.trade_count * 100
        avg_return = g.total_return / g.trade_count * 100
        
        log.info(f"=== 策略统计 ===")
        log.info(f"总交易次数: {g.trade_count}")
        log.info(f"胜率: {win_rate:.2f}%")
        log.info(f"平均收益率: {avg_return:.2f}%")


def before_trading_start(context):
    """
    开盘前运行函数
    """
    pass


def after_trading_end(context):
    """
    收盘后运行函数
    """
    pass


# ===== 辅助函数 =====

def get_stock_info(stock_code):
    """
    获取股票基本信息
    """
    try:
        security_info = get_security_info(stock_code)
        return {
            'name': security_info.display_name,
            'start_date': security_info.start_date,
            'end_date': security_info.end_date
        }
    except:
        return None


def is_st_stock(stock_code):
    """
    判断是否为ST股票
    """
    try:
        security_info = get_security_info(stock_code)
        return 'ST' in security_info.display_name or '*ST' in security_info.display_name
    except:
        return True


def get_trading_days_between(start_date, end_date):
    """
    获取两个日期之间的交易日数
    """
    try:
        trading_days = get_trade_days(start_date=start_date, end_date=end_date)
        return len(trading_days) - 1  # 不包含起始日
    except:
        return 0


def calculate_limit_price(stock_code, base_price, is_up=True):
    """
    计算涨跌停价格
    """
    # 根据股票代码判断板块
    if stock_code.startswith('688') or stock_code.startswith('689'):
        # 科创板：20%
        limit_ratio = 0.20
    elif stock_code.startswith('300') or stock_code.startswith('301'):
        # 创业板：20%
        limit_ratio = 0.20
    else:
        # 主板：10%
        limit_ratio = 0.10
    
    if is_up:
        return base_price * (1 + limit_ratio)
    else:
        return base_price * (1 - limit_ratio)


def log_error_and_continue(func_name, error, stock_code=None):
    """
    记录错误并继续执行
    """
    error_msg = f"{func_name} 执行出错"
    if stock_code:
        error_msg += f" (股票: {stock_code})"
    error_msg += f": {error}"
    log.error(error_msg)


# ===== 策略参数配置 =====

class StrategyConfig:
    """策略配置类"""
    
    # 股票池配置
    MIN_MARKET_CAP = 50 * 100000000  # 最小流通市值50亿
    MAX_MARKET_CAP = 200 * 100000000  # 最大流通市值200亿
    MIN_LIST_DAYS = 60  # 最小上市天数
    TRACK_DAYS = 5  # 跟踪天数
    
    # 买入配置
    VALID_BUY_DAYS = [2, 4]  # 有效买入日
    MA5_TOLERANCE = 0.02  # MA5容差
    MAX_POSITION_RATIO = 0.10  # 最大单股仓位
    SLIPPAGE_RATIO = 0.02  # 滑点
    
    # 卖出配置
    MAX_HOLD_DAYS = 15  # 最大持有天数
    MA5_BREAKDOWN_THRESHOLD = -0.004  # MA5跌破阈值
    HIGH_DROP_THRESHOLD = 0.09  # 冲高回落阈值
    
    # 仓位管理
    MAX_POSITIONS = 10  # 最大持仓数量


# ===== 策略说明 =====
"""
策略核心逻辑说明：

1. 股票池选择模块 (joinquant_stock_pool.py)：
   - 每日筛选首板涨停股票
   - 过滤条件：非ST股票、上市满60天、流通市值50-200亿
   - 建立5日动态跟踪池，记录首板日期和价格

2. 买入时机模块 (joinquant_buy_timing.py)：
   - 买入条件：首板后第2天或第4天
   - 价格条件：回踩至MA5附近（±2%范围）
   - 风控条件：当日未跌停、仓位未满
   - 执行方式：市价单买入，单股仓位不超过10%

3. 卖出时机模块 (joinquant_sell_timing.py)：
   - 止盈条件：收盘价跌破MA5、持仓超过15天
   - 止损条件：跌停后第二天、炸板（冲高9%后收盘未涨停）
   - 特殊规则：涨停日不卖出、断板止盈
   - 执行方式：限价单卖出

4. 风险控制：
   - 最大持仓10只股票
   - 单股仓位不超过总资金10%
   - 设置2%滑点容差
   - 严格的买卖条件判断

5. 策略优势：
   - 基于成熟实盘策略改编
   - 完整的风险控制体系
   - 模块化设计便于维护
   - 详细的日志记录和统计

使用说明：
1. 将所有模块文件放在同一目录下
2. 在聚宽平台创建新策略
3. 复制本文件内容到策略编辑器
4. 设置回测参数并运行

注意事项：
- 策略适用于A股市场
- 建议回测期间至少6个月以上
- 可根据实际情况调整参数配置
- 实盘使用前请充分回测验证
"""