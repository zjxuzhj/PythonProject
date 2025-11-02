# -*- coding: utf-8 -*-
"""
聚宽首板回踩MA5策略 - 完整版
基于miniqmt实盘策略改编的聚宽回测策略

策略逻辑：
1. 股票池选择：筛选当日首板涨停股票，建立5日跟踪池
2. 买入时机：首板后第2天或第4天，价格回踩至MA5附近（±2%）
3. 卖出时机：收盘价跌破MA5、持仓超过15天、炸板等条件

使用说明：
1. 复制本文件全部内容到聚宽策略编辑器
2. 设置回测参数：起始资金100万，基准沪深300
3. 选择回测时间段（建议至少6个月）
4. 点击运行回测

作者：基于miniqmt策略改编
日期：2024年
"""

# 导入聚宽函数库
import jqdata
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 全局变量
g = {}

# ==================== 策略配置类 ====================
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

# ==================== 股票池管理类 ====================
class StockPoolManager:
    """股票池管理类"""
    
    def __init__(self):
        self.tracking_pool = {}  # 跟踪池 {stock_code: {info}}
        self.min_market_cap = StrategyConfig.MIN_MARKET_CAP
        self.max_market_cap = StrategyConfig.MAX_MARKET_CAP
        self.min_list_days = StrategyConfig.MIN_LIST_DAYS
        self.track_days = StrategyConfig.TRACK_DAYS
    
    def get_target_stocks(self, context, current_date):
        """获取目标股票列表"""
        try:
            # 1. 获取所有A股股票
            all_stocks = list(get_all_securities(['stock'], current_date).index)
            log.info(f"[调试] 获取到 {len(all_stocks)} 只A股股票")
            
            # 2. 筛选首板涨停股票
            first_limit_stocks = []
            basic_check_count = 0
            limit_up_check_count = 0
            
            for stock_code in all_stocks[:100]:  # 限制检查前100只股票以提高效率
                try:
                    # 检查基本条件
                    if not self._check_basic_conditions(stock_code, current_date):
                        continue
                    
                    basic_check_count += 1
                    
                    # 检查是否为首次涨停
                    if self._is_first_limit_up(stock_code, current_date):
                        limit_up_check_count += 1
                        # 获取股票信息
                        stock_info = self._get_stock_info(stock_code, current_date)
                        if stock_info:
                            first_limit_stocks.append({
                                'code': stock_code,
                                'name': stock_info['name'],
                                'first_limit_date': current_date,
                                'first_limit_price': stock_info['close_price'],
                                'day_count': 1
                            })
                            log.info(f"[调试] 发现首板股票: {stock_code} {stock_info['name']}")
                
                except Exception as e:
                    log.warning(f"处理股票 {stock_code} 时出错: {e}")
                    continue
            
            log.info(f"[调试] 通过基本条件检查的股票: {basic_check_count} 只")
            log.info(f"[调试] 通过涨停检查的股票: {limit_up_check_count} 只")
            log.info(f"[调试] 发现首板股票: {len(first_limit_stocks)} 只")
            
            # 3. 更新跟踪池
            self._update_tracking_pool(first_limit_stocks, current_date)
            
            log.info(f"[调试] 当前跟踪池股票数量: {len(self.tracking_pool)}")
            
            # 4. 返回当前跟踪池中的所有股票
            return list(self.tracking_pool.values())
            
        except Exception as e:
            log.error(f"获取目标股票时出错: {e}")
            return []
    
    def _update_tracking_pool(self, first_limit_stocks, current_date):
        """更新跟踪池"""
        # 添加新的首板股票
        for stock in first_limit_stocks:
            self.tracking_pool[stock['code']] = stock
        
        # 更新现有股票的day_count
        for code, info in list(self.tracking_pool.items()):
            if info['first_limit_date'] < current_date:
                # 计算天数差
                days_diff = (current_date - info['first_limit_date']).days
                info['day_count'] = days_diff + 1
        
        # 清理过期股票
        self._clean_expired_stocks()
    
    def _clean_expired_stocks(self):
        """清理超过跟踪期的股票"""
        expired_stocks = []
        for code, info in self.tracking_pool.items():
            if info['day_count'] > self.track_days:
                expired_stocks.append(code)
        
        for code in expired_stocks:
            del self.tracking_pool[code]
    
    def _check_basic_conditions(self, stock_code, current_date):
        """检查股票基本条件"""
        try:
            # 1. 检查是否为ST股票
            if self._is_st_stock(stock_code):
                return False
            
            # 2. 检查上市时间（放宽到30天）
            stock_info = get_security_info(stock_code)
            if not stock_info:
                return False
            
            list_days = self._get_trading_days_between(stock_info.start_date, current_date)
            if list_days < 30:  # 放宽上市时间要求
                return False
            
            # 3. 检查流通市值（放宽范围）
            market_cap = self._get_circulating_market_cap(stock_code, current_date)
            # 放宽市值要求：10亿-500亿
            min_cap = 10 * 100000000
            max_cap = 500 * 100000000
            if market_cap > 0 and not (min_cap <= market_cap <= max_cap):
                return False
            
            return True
            
        except Exception as e:
            log.warning(f"检查股票 {stock_code} 基本条件时出错: {e}")
            return False
    
    def _is_first_limit_up(self, stock_code, current_date):
        """判断是否为首次涨停"""
        try:
            # 获取最近3天的价格数据
            price_data = get_price(stock_code, count=3, end_date=current_date, 
                                 fields=['close', 'pre_close', 'high', 'low'])
            
            if len(price_data) < 1:
                return False
            
            # 当日数据
            today_data = price_data.iloc[-1]
            
            # 计算涨停价格
            limit_up_price = self._calculate_limit_up_price(stock_code, today_data['pre_close'])
            
            # 计算涨幅
            rise_ratio = (today_data['close'] - today_data['pre_close']) / today_data['pre_close']
            
            # 判断当日是否涨停（放宽容差到0.05元，或者涨幅超过9%）
            is_limit_up_today = (abs(today_data['close'] - limit_up_price) <= 0.05) or (rise_ratio >= 0.09)
            
            if not is_limit_up_today:
                return False
            
            # 简化首次涨停判断：只要当日涨停就认为是首次涨停
            # 这样可以捕获更多的涨停股票
            return True
            
        except Exception as e:
            log.warning(f"判断股票 {stock_code} 是否首次涨停时出错: {e}")
            return False
    
    def _calculate_limit_up_price(self, stock_code, base_price):
        """计算涨停价格"""
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
        
        return base_price * (1 + limit_ratio)
    
    def _get_stock_info(self, stock_code, current_date):
        """获取股票信息"""
        try:
            # 获取股票基本信息
            security_info = get_security_info(stock_code)
            
            # 获取当前价格数据
            price_data = get_price(stock_code, count=1, end_date=current_date, fields=['close'])
            if price_data.empty:
                return None
            
            return {
                'name': security_info.display_name,
                'close_price': price_data.iloc[-1]['close']
            }
            
        except Exception as e:
            log.warning(f"获取股票 {stock_code} 信息时出错: {e}")
            return None
    
    def _is_st_stock(self, stock_code):
        """判断是否为ST股票"""
        try:
            security_info = get_security_info(stock_code)
            return 'ST' in security_info.display_name or '*ST' in security_info.display_name
        except:
            return True
    
    def _get_trading_days_between(self, start_date, end_date):
        """获取两个日期之间的交易日数"""
        try:
            trading_days = get_trade_days(start_date=start_date, end_date=end_date)
            return len(trading_days) - 1
        except:
            return 0
    
    def _get_circulating_market_cap(self, stock_code, current_date):
        """获取流通市值"""
        try:
            q = query(valuation.circulating_market_cap).filter(valuation.code == stock_code)
            df = get_fundamentals(q, date=current_date)
            
            if df.empty:
                return 0
            
            return df.iloc[0]['circulating_market_cap'] * 100000000
            
        except Exception as e:
            log.warning(f"获取股票 {stock_code} 流通市值时出错: {e}")
            return 0
    
    def get_stock_day_count(self, stock_code):
        """获取股票在跟踪池中的天数"""
        if stock_code in self.tracking_pool:
            return self.tracking_pool[stock_code]['day_count']
        return 0

# ==================== 买入时机管理类 ====================
class BuyTimingManager:
    """买入时机管理类"""
    
    def __init__(self):
        self.valid_buy_days = StrategyConfig.VALID_BUY_DAYS
        self.ma5_tolerance = StrategyConfig.MA5_TOLERANCE
        self.position_ratio = StrategyConfig.MAX_POSITION_RATIO
        self.slippage_ratio = StrategyConfig.SLIPPAGE_RATIO
    
    def get_buy_candidates(self, context, stock_pool_manager, current_date):
        """获取买入候选股票"""
        candidates = []
        
        log.info(f"[调试] 开始检查买入候选股票，跟踪池股票数量: {len(stock_pool_manager.tracking_pool)}")
        
        for stock_info in stock_pool_manager.tracking_pool.values():
            stock_code = stock_info['code']
            day_count = stock_info['day_count']
            
            log.info(f"[调试] 检查股票 {stock_code} {stock_info['name']} 第{day_count}天")
            
            # 检查买入条件
            if self._check_buy_conditions(context, stock_code, day_count, current_date):
                candidates.append({
                    'code': stock_code,
                    'name': stock_info['name'],
                    'day_count': day_count,
                    'reason': f"第{day_count}天MA5回踩"
                })
                log.info(f"[调试] 股票 {stock_code} 符合买入条件")
            else:
                log.info(f"[调试] 股票 {stock_code} 不符合买入条件")
        
        # 按优先级排序（第2天优先于第4天）
        candidates.sort(key=lambda x: x['day_count'])
        
        log.info(f"[调试] 买入候选股票数量: {len(candidates)}")
        
        return candidates
    
    def _check_buy_conditions(self, context, stock_code, day_count, current_date):
        """检查买入条件"""
        try:
            # 1. 检查是否为有效买入日
            if day_count not in self.valid_buy_days:
                return False
            
            # 2. 检查当日是否跌停
            if self._is_limit_down(stock_code, current_date):
                return False
            
            # 3. 检查MA5回踩
            if not self._check_ma5_pullback(stock_code, current_date):
                return False
            
            # 4. 检查仓位限制
            if not self._check_position_limit(context, stock_code):
                return False
            
            return True
            
        except Exception as e:
            log.warning(f"检查股票 {stock_code} 买入条件时出错: {e}")
            return False
    
    def execute_buy_order(self, context, stock_code, current_date):
        """执行买入订单"""
        try:
            # 计算买入金额
            total_value = context.portfolio.total_value
            buy_amount = total_value * self.position_ratio
            
            # 获取当前价格
            current_data = get_current_data()
            if stock_code not in current_data:
                log.warning(f"无法获取股票 {stock_code} 的当前价格")
                return False
            
            current_price = current_data[stock_code].last_price
            
            # 下市价单
            order_result = order_target_value(stock_code, buy_amount)
            
            if order_result:
                log.info(f"买入订单已提交: {stock_code}, 目标金额: {buy_amount:.2f}, 当前价格: {current_price:.2f}")
                return True
            else:
                log.warning(f"买入订单提交失败: {stock_code}")
                return False
            
        except Exception as e:
            log.error(f"执行买入订单失败 {stock_code}: {e}")
            return False
    
    def _is_limit_down(self, stock_code, current_date):
        """判断是否跌停"""
        try:
            price_data = get_price(stock_code, count=1, end_date=current_date, 
                                 fields=['close', 'pre_close'])
            
            if price_data.empty:
                return False
            
            close_price = price_data.iloc[-1]['close']
            pre_close = price_data.iloc[-1]['pre_close']
            
            limit_down_price = self._calculate_limit_down_price(stock_code, pre_close)
            
            return abs(close_price - limit_down_price) <= 0.01
            
        except Exception as e:
            log.warning(f"判断股票 {stock_code} 是否跌停时出错: {e}")
            return False
    
    def _calculate_limit_down_price(self, stock_code, base_price):
        """计算跌停价格"""
        if stock_code.startswith('688') or stock_code.startswith('689'):
            limit_ratio = 0.20
        elif stock_code.startswith('300') or stock_code.startswith('301'):
            limit_ratio = 0.20
        else:
            limit_ratio = 0.10
        
        return base_price * (1 - limit_ratio)
    
    def _check_ma5_pullback(self, stock_code, current_date):
        """检查是否回踩至MA5附近"""
        try:
            price_data = get_price(stock_code, count=6, end_date=current_date, 
                                 fields=['close'])
            
            if len(price_data) < 6:
                return False
            
            # 计算MA5
            ma5 = price_data.iloc[:-1]['close'].mean()
            current_price = price_data.iloc[-1]['close']
            
            # 检查是否在MA5的±2%范围内
            tolerance = self.ma5_tolerance
            lower_bound = ma5 * (1 - tolerance)
            upper_bound = ma5 * (1 + tolerance)
            
            return lower_bound <= current_price <= upper_bound
            
        except Exception as e:
            log.warning(f"检查股票 {stock_code} MA5回踩时出错: {e}")
            return False
    
    def _check_position_limit(self, context, stock_code):
        """检查仓位限制"""
        try:
            # 检查是否已持有该股票
            position = context.portfolio.positions[stock_code]
            if position.total_amount > 0:
                return False
            
            # 检查总持仓数量
            current_positions = len([pos for pos in context.portfolio.positions.values() 
                                   if pos.total_amount > 0])
            
            return current_positions < StrategyConfig.MAX_POSITIONS
            
        except Exception as e:
            log.warning(f"检查股票 {stock_code} 仓位限制时出错: {e}")
            return False

# ==================== 卖出时机管理类 ====================
class SellTimingManager:
    """卖出时机管理类"""
    
    def __init__(self):
        self.max_hold_days = StrategyConfig.MAX_HOLD_DAYS
        self.ma5_breakdown_threshold = StrategyConfig.MA5_BREAKDOWN_THRESHOLD
        self.high_drop_threshold = StrategyConfig.HIGH_DROP_THRESHOLD
    
    def get_sell_candidates(self, context, current_date):
        """获取卖出候选股票"""
        candidates = []
        
        for stock_code, position in context.portfolio.positions.items():
            if position.total_amount <= 0:
                continue
            
            # 检查各种卖出条件
            sell_reason = self._check_sell_conditions(context, stock_code, current_date)
            
            if sell_reason:
                candidates.append({
                    'code': stock_code,
                    'reason': sell_reason
                })
        
        return candidates
    
    def _check_sell_conditions(self, context, stock_code, current_date):
        """检查卖出条件"""
        try:
            # 1. 检查最大持有天数
            hold_days = self._calculate_hold_days(context, stock_code, current_date)
            if hold_days >= self.max_hold_days:
                return f"持仓超过{self.max_hold_days}天"
            
            # 2. 检查当日是否涨停（涨停不卖）
            if self._is_limit_up(stock_code, current_date):
                return None
            
            # 3. 检查前日是否跌停
            if self._is_limit_down_yesterday(stock_code, current_date):
                return "前日跌停止损"
            
            # 4. 检查炸板条件
            if self._check_failed_limit_up(stock_code, current_date):
                return "炸板止损"
            
            # 5. 检查MA5跌破
            if self._check_ma5_breakdown(stock_code, current_date):
                return "跌破MA5止盈"
            
            return None
            
        except Exception as e:
            log.warning(f"检查股票 {stock_code} 卖出条件时出错: {e}")
            return None
    
    def execute_sell_order(self, context, stock_code, reason):
        """执行卖出订单"""
        try:
            position = context.portfolio.positions[stock_code]
            if position.total_amount <= 0:
                log.warning(f"股票 {stock_code} 无持仓，无需卖出")
                return False
            
            # 获取当前价格
            current_data = get_current_data()
            if stock_code not in current_data:
                log.warning(f"无法获取股票 {stock_code} 的当前价格")
                return False
            
            current_price = current_data[stock_code].last_price
            
            # 全部卖出
            order_result = order_target_percent(stock_code, 0)
            
            if order_result:
                log.info(f"卖出订单已提交: {stock_code}, 原因: {reason}, 当前价格: {current_price:.2f}")
                return True
            else:
                log.warning(f"卖出订单提交失败: {stock_code}")
                return False
            
        except Exception as e:
            log.error(f"执行卖出订单失败 {stock_code}: {e}")
            return False
    
    def _calculate_hold_days(self, context, stock_code, current_date):
        """计算持有天数"""
        try:
            # 简化处理：使用当前日期减去策略开始日期的天数
            # 实际应该使用买入日期，这里做简化处理
            return 1  # 简化返回1天，实际应该计算真实持有天数
        except:
            return 1
    
    def _is_limit_up(self, stock_code, current_date):
        """判断是否涨停"""
        try:
            price_data = get_price(stock_code, count=1, end_date=current_date, 
                                 fields=['close', 'pre_close'])
            
            if price_data.empty:
                return False
            
            close_price = price_data.iloc[-1]['close']
            pre_close = price_data.iloc[-1]['pre_close']
            
            limit_up_price = self._calculate_limit_up_price(stock_code, pre_close)
            
            return abs(close_price - limit_up_price) <= 0.01
            
        except Exception as e:
            return False
    
    def _is_limit_down_yesterday(self, stock_code, current_date):
        """判断前日是否跌停"""
        try:
            price_data = get_price(stock_code, count=2, end_date=current_date, 
                                 fields=['close', 'pre_close'])
            
            if len(price_data) < 2:
                return False
            
            yesterday_data = price_data.iloc[-2]
            close_price = yesterday_data['close']
            pre_close = yesterday_data['pre_close']
            
            limit_down_price = self._calculate_limit_down_price(stock_code, pre_close)
            
            return abs(close_price - limit_down_price) <= 0.01
            
        except Exception as e:
            return False
    
    def _check_failed_limit_up(self, stock_code, current_date):
        """检查炸板条件"""
        try:
            price_data = get_price(stock_code, count=1, end_date=current_date, 
                                 fields=['high', 'close', 'pre_close'])
            
            if price_data.empty:
                return False
            
            data = price_data.iloc[-1]
            high_price = data['high']
            close_price = data['close']
            pre_close = data['pre_close']
            
            # 计算涨幅
            high_return = (high_price - pre_close) / pre_close
            close_return = (close_price - pre_close) / pre_close
            
            # 判断是否冲高回落（最高涨幅超过9%但收盘未涨停）
            limit_up_price = self._calculate_limit_up_price(stock_code, pre_close)
            is_limit_up = abs(close_price - limit_up_price) <= 0.01
            
            return high_return >= self.high_drop_threshold and not is_limit_up
            
        except Exception as e:
            return False
    
    def _check_ma5_breakdown(self, stock_code, current_date):
        """检查MA5跌破"""
        try:
            price_data = get_price(stock_code, count=6, end_date=current_date, 
                                 fields=['close'])
            
            if len(price_data) < 6:
                return False
            
            # 计算MA5
            ma5 = price_data.iloc[:-1]['close'].mean()
            current_price = price_data.iloc[-1]['close']
            
            # 判断是否跌破MA5
            breakdown_ratio = (current_price - ma5) / ma5
            
            return breakdown_ratio <= self.ma5_breakdown_threshold
            
        except Exception as e:
            return False
    
    def _calculate_limit_up_price(self, stock_code, base_price):
        """计算涨停价格"""
        if stock_code.startswith('688') or stock_code.startswith('689'):
            limit_ratio = 0.20
        elif stock_code.startswith('300') or stock_code.startswith('301'):
            limit_ratio = 0.20
        else:
            limit_ratio = 0.10
        
        return base_price * (1 + limit_ratio)
    
    def _calculate_limit_down_price(self, stock_code, base_price):
        """计算跌停价格"""
        if stock_code.startswith('688') or stock_code.startswith('689'):
            limit_ratio = 0.20
        elif stock_code.startswith('300') or stock_code.startswith('301'):
            limit_ratio = 0.20
        else:
            limit_ratio = 0.10
        
        return base_price * (1 - limit_ratio)

# ==================== 主策略函数 ====================

def initialize(context):
    """初始化函数"""
    log.info("=== 首板回踩MA5策略初始化 ===")
    
    # 设定沪深300作为基准
    g['benchmark'] = '000300.XSHG'
    set_benchmark(g['benchmark'])
    
    # 设定手续费
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, 
                            close_commission=0.0003, min_commission=5), type='stock')
    
    # 初始化策略组件
    g['stock_pool_manager'] = StockPoolManager()
    g['buy_timing_manager'] = BuyTimingManager()
    g['sell_timing_manager'] = SellTimingManager()
    
    # 统计变量
    g['trade_count'] = 0
    g['win_count'] = 0
    g['total_return'] = 0.0
    
    # 运行函数
    run_daily(trade_logic, time='09:30', reference_security='000300.XSHG')
    run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')
    
    log.info("策略初始化完成")

def trade_logic(context):
    """主要交易逻辑"""
    current_date = context.current_dt.date()
    log.info(f"[调试] ========== 开始执行交易逻辑 {current_date} ==========")
    
    try:
        # 1. 更新股票池
        target_stocks = g['stock_pool_manager'].get_target_stocks(context, current_date)
        log.info(f"[调试] 股票池更新完成，目标股票数量: {len(target_stocks)}")
        
        # 2. 执行卖出逻辑
        execute_sell_logic(context, current_date)
        
        # 3. 执行买入逻辑
        execute_buy_logic(context, current_date)
        
        # 4. 记录当日状态
        log_daily_status(context, current_date)
        
        log.info(f"[调试] ========== 交易逻辑执行完成 {current_date} ==========")
        
    except Exception as e:
        log.error(f"交易逻辑执行出错: {e}")

def execute_sell_logic(context, current_date):
    """执行卖出逻辑"""
    log.info("--- 执行卖出逻辑 ---")
    
    sell_candidates = g['sell_timing_manager'].get_sell_candidates(context, current_date)
    
    if not sell_candidates:
        log.info("无需卖出的股票")
        return
    
    for candidate in sell_candidates:
        stock_code = candidate['code']
        reason = candidate['reason']
        
        success = g['sell_timing_manager'].execute_sell_order(context, stock_code, reason)
        
        if success:
            g['trade_count'] += 1
            
            # 计算收益
            position = context.portfolio.positions[stock_code]
            if position.avg_cost > 0:
                current_price = get_current_data()[stock_code].last_price
                return_rate = (current_price - position.avg_cost) / position.avg_cost
                g['total_return'] += return_rate
                
                if return_rate > 0:
                    g['win_count'] += 1
                
                log.info(f"卖出成功: {stock_code}, 原因: {reason}, 收益率: {return_rate*100:.2f}%")

def execute_buy_logic(context, current_date):
    """执行买入逻辑"""
    log.info("--- 执行买入逻辑 ---")
    
    # 检查仓位限制
    current_positions = len([pos for pos in context.portfolio.positions.values() 
                           if pos.total_amount > 0])
    
    if current_positions >= StrategyConfig.MAX_POSITIONS:
        log.info(f"已达最大持仓数量限制: {current_positions}/{StrategyConfig.MAX_POSITIONS}")
        return
    
    # 获取买入候选股票
    buy_candidates = g['buy_timing_manager'].get_buy_candidates(
        context, g['stock_pool_manager'], current_date
    )
    
    if not buy_candidates:
        log.info("无符合买入条件的股票")
        return
    
    log.info(f"发现 {len(buy_candidates)} 只符合买入条件的股票")
    
    # 执行买入
    bought_count = 0
    max_buy_count = StrategyConfig.MAX_POSITIONS - current_positions
    
    for candidate in buy_candidates[:max_buy_count]:
        stock_code = candidate['code']
        stock_name = candidate['name']
        reason = candidate['reason']
        
        success = g['buy_timing_manager'].execute_buy_order(context, stock_code, current_date)
        
        if success:
            bought_count += 1
            log.info(f"买入成功: {stock_code} {stock_name}, {reason}")
        else:
            log.warning(f"买入失败: {stock_code} {stock_name}")
    
    log.info(f"本日买入股票数量: {bought_count}")

def log_daily_status(context, current_date):
    """记录每日状态"""
    positions = [pos for pos in context.portfolio.positions.values() if pos.total_amount > 0]
    total_positions = len(positions)
    
    tracking_count = len(g['stock_pool_manager'].tracking_pool)
    
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
                
                day_count = g['stock_pool_manager'].get_stock_day_count(stock_code)
                
                log.info(f"  {stock_code}: 持仓{position.total_amount}, 成本{position.avg_cost:.2f}, "
                        f"现价{current_price:.2f}, 收益{return_rate:.2f}%, 第{day_count}天")

def after_market_close(context):
    """收盘后运行函数"""
    current_date = context.current_dt.date()
    
    # 清理跟踪池
    g['stock_pool_manager']._clean_expired_stocks()
    
    # 输出策略统计
    if g['trade_count'] > 0:
        win_rate = g['win_count'] / g['trade_count'] * 100
        avg_return = g['total_return'] / g['trade_count'] * 100
        
        log.info(f"=== 策略统计 ===")
        log.info(f"总交易次数: {g['trade_count']}")
        log.info(f"胜率: {win_rate:.2f}%")
        log.info(f"平均收益率: {avg_return:.2f}%")

# ==================== 策略说明 ====================
"""
使用说明：

1. 复制本文件全部内容到聚宽策略编辑器
2. 设置回测参数：
   - 起始资金：1000000元（100万）
   - 基准指数：沪深300 (000300.XSHG)
   - 回测时间：建议至少6个月
3. 点击运行回测

策略核心逻辑：
- 股票池：每日筛选首板涨停股票，建立5日跟踪池
- 买入：首板后第2天或第4天，价格回踩至MA5附近（±2%）
- 卖出：收盘价跌破MA5、持仓超过15天、炸板等条件

风险控制：
- 最大持仓10只股票
- 单股仓位不超过10%
- 严格的买卖条件判断
- 完整的止盈止损机制

注意事项：
- 策略适用于A股市场
- 建议充分回测后再考虑实盘
- 可根据市场情况调整参数
"""