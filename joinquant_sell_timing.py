# -*- coding: utf-8 -*-
"""
聚宽策略 - 卖出时机模块
基于止盈止损条件的卖出判断和执行逻辑
"""

import jqdata
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional


class SellTimingManager:
    """卖出时机管理器 - 负责卖出条件判断和执行"""
    
    def __init__(self):
        # 配置参数
        self.max_hold_days = 15  # 最大持有天数
        self.ma5_breakdown_threshold = -0.004  # 跌破MA5的阈值（-0.4%）
        self.high_drop_threshold = 0.09  # 冲高回落阈值（9%）
        
    def check_sell_conditions(self, context, stock_code: str, current_date) -> Tuple[bool, str]:
        """
        检查卖出条件
        
        Args:
            context: 聚宽上下文对象
            stock_code: 股票代码
            current_date: 当前日期
            
        Returns:
            (是否应该卖出, 卖出原因)
        """
        try:
            # 获取持仓信息
            if stock_code not in context.portfolio.positions:
                return False, "未持有该股票"
            
            position = context.portfolio.positions[stock_code]
            if position.total_amount <= 0:
                return False, "持仓为0"
            
            # 计算持有天数
            hold_days = self._calculate_hold_days(context, stock_code, current_date)
            
            # 1. 检查最大持有天数限制
            if hold_days >= self.max_hold_days:
                return True, f"持有超限({hold_days}天)"
            
            # 2. 检查当日是否涨停（涨停不卖）
            if self._is_limit_up(stock_code, current_date):
                return False, "当日涨停"
            
            # 3. 检查前一日是否跌停（跌停后第二天卖出）
            if self._is_prev_limit_down(stock_code, current_date):
                return True, "昨日跌停止损"
            
            # 4. 检查前一日是否涨停（断板止盈）
            if self._is_prev_limit_up(stock_code, current_date):
                return True, "断板止盈"
            
            # 5. 检查炸板卖出条件
            bomb_sell, bomb_reason = self._check_bomb_sell(stock_code, current_date)
            if bomb_sell:
                return True, bomb_reason
            
            # 6. 检查收盘价是否跌破MA5
            ma5_sell, ma5_reason = self._check_ma5_breakdown(stock_code, current_date)
            if ma5_sell:
                return True, ma5_reason
            
            return False, "持有"
            
        except Exception as e:
            log.error(f"检查股票 {stock_code} 卖出条件时出错: {e}")
            return True, f"检查出错，保守卖出: {e}"
    
    def execute_sell_order(self, context, stock_code: str, reason: str) -> bool:
        """
        执行卖出订单
        
        Args:
            context: 聚宽上下文对象
            stock_code: 股票代码
            reason: 卖出原因
            
        Returns:
            是否成功下单
        """
        try:
            # 获取当前持仓
            if stock_code not in context.portfolio.positions:
                return False
            
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
            
            # 全部卖出（使用order_target_percent设置为0%）
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
    
    def _calculate_hold_days(self, context, stock_code: str, current_date) -> int:
        """
        计算持有天数
        
        Args:
            context: 聚宽上下文对象
            stock_code: 股票代码
            current_date: 当前日期
            
        Returns:
            持有天数
        """
        try:
            # 从交易记录中获取买入日期
            # 这里简化处理，实际应该从交易记录中获取
            # 可以通过context.portfolio.positions[stock_code]的相关属性获取
            
            # 获取最近的交易记录
            orders = get_orders()
            buy_date = None
            
            for order in orders.values():
                if (order.security == stock_code and 
                    order.side == 'long' and 
                    order.filled > 0):
                    buy_date = order.add_time.date()
                    break
            
            if buy_date:
                # 计算交易日差
                trading_days = get_trade_days(start_date=buy_date, end_date=current_date)
                return len(trading_days) - 1
            else:
                return 0
                
        except Exception as e:
            log.error(f"计算持有天数时出错: {e}")
            return 0
    
    def _is_limit_up(self, stock_code: str, current_date) -> bool:
        """
        检查当日是否涨停
        
        Args:
            stock_code: 股票代码
            current_date: 当前日期
            
        Returns:
            是否涨停
        """
        try:
            # 获取当日和前一日价格数据
            price_data = get_price(
                stock_code,
                start_date=current_date - timedelta(days=2),
                end_date=current_date,
                frequency='daily',
                fields=['close']
            )
            
            if len(price_data) < 2:
                return False
            
            prev_close = price_data.iloc[-2]['close']
            current_close = price_data.iloc[-1]['close']
            
            # 计算涨停价
            limit_up_price = self._calculate_limit_up_price(stock_code, prev_close)
            
            # 判断是否涨停（允许0.01的误差）
            return abs(current_close - limit_up_price) < 0.01
            
        except Exception as e:
            log.error(f"检查股票 {stock_code} 是否涨停时出错: {e}")
            return False
    
    def _is_prev_limit_down(self, stock_code: str, current_date) -> bool:
        """
        检查前一日是否跌停
        
        Args:
            stock_code: 股票代码
            current_date: 当前日期
            
        Returns:
            前一日是否跌停
        """
        try:
            # 获取前两日价格数据
            price_data = get_price(
                stock_code,
                start_date=current_date - timedelta(days=3),
                end_date=current_date - timedelta(days=1),
                frequency='daily',
                fields=['close']
            )
            
            if len(price_data) < 2:
                return False
            
            prev_prev_close = price_data.iloc[-2]['close']
            prev_close = price_data.iloc[-1]['close']
            
            # 计算跌停价
            limit_down_price = self._calculate_limit_down_price(stock_code, prev_prev_close)
            
            # 判断前一日是否跌停
            return abs(prev_close - limit_down_price) < 0.01
            
        except Exception as e:
            log.error(f"检查股票 {stock_code} 前一日是否跌停时出错: {e}")
            return False
    
    def _is_prev_limit_up(self, stock_code: str, current_date) -> bool:
        """
        检查前一日是否涨停
        
        Args:
            stock_code: 股票代码
            current_date: 当前日期
            
        Returns:
            前一日是否涨停
        """
        try:
            # 获取前两日价格数据
            price_data = get_price(
                stock_code,
                start_date=current_date - timedelta(days=3),
                end_date=current_date - timedelta(days=1),
                frequency='daily',
                fields=['close']
            )
            
            if len(price_data) < 2:
                return False
            
            prev_prev_close = price_data.iloc[-2]['close']
            prev_close = price_data.iloc[-1]['close']
            
            # 计算涨停价
            limit_up_price = self._calculate_limit_up_price(stock_code, prev_prev_close)
            
            # 判断前一日是否涨停
            return abs(prev_close - limit_up_price) < 0.01
            
        except Exception as e:
            log.error(f"检查股票 {stock_code} 前一日是否涨停时出错: {e}")
            return False
    
    def _check_bomb_sell(self, stock_code: str, current_date) -> Tuple[bool, str]:
        """
        检查炸板卖出条件：当天最高价超过9%但收盘跌破9%
        
        Args:
            stock_code: 股票代码
            current_date: 当前日期
            
        Returns:
            (是否炸板卖出, 原因说明)
        """
        try:
            # 获取当日和前一日价格数据
            price_data = get_price(
                stock_code,
                start_date=current_date - timedelta(days=2),
                end_date=current_date,
                frequency='daily',
                fields=['close', 'high', 'low', 'open']
            )
            
            if len(price_data) < 2:
                return False, "数据不足"
            
            prev_close = price_data.iloc[-2]['close']
            current_data = price_data.iloc[-1]
            current_high = current_data['high']
            current_close = current_data['close']
            
            # 计算9%涨幅价格
            price_9_percent = prev_close * 1.09
            
            # 检查是否冲高到9%以上
            is_high_touched = current_high >= price_9_percent
            
            # 检查收盘是否跌破9%
            is_close_below_9 = current_close < price_9_percent
            
            # 检查是否未涨停收盘
            limit_up_price = self._calculate_limit_up_price(stock_code, prev_close)
            is_not_limit_close = current_close < limit_up_price
            
            if is_high_touched and is_close_below_9 and is_not_limit_close:
                return True, f"炸板卖出(冲高{(current_high/prev_close-1)*100:.1f}%,收盘{(current_close/prev_close-1)*100:.1f}%)"
            
            return False, "未炸板"
            
        except Exception as e:
            log.error(f"检查股票 {stock_code} 炸板条件时出错: {e}")
            return False, f"炸板检查出错: {e}"
    
    def _check_ma5_breakdown(self, stock_code: str, current_date) -> Tuple[bool, str]:
        """
        检查收盘价是否跌破MA5
        
        Args:
            stock_code: 股票代码
            current_date: 当前日期
            
        Returns:
            (是否跌破MA5, 原因说明)
        """
        try:
            # 获取最近10个交易日的数据用于计算MA5
            price_data = get_price(
                stock_code,
                start_date=current_date - timedelta(days=20),
                end_date=current_date,
                frequency='daily',
                fields=['close']
            )
            
            if len(price_data) < 5:
                return False, "数据不足计算MA5"
            
            # 计算MA5
            price_data['ma5'] = price_data['close'].rolling(window=5).mean()
            
            # 获取当日数据
            current_data = price_data.iloc[-1]
            current_close = current_data['close']
            current_ma5 = current_data['ma5']
            
            if pd.isna(current_ma5):
                return False, "MA5计算失败"
            
            # 计算跌破MA5的幅度
            breakdown_ratio = (current_close - current_ma5) / current_ma5
            
            # 检查是否跌破MA5阈值
            if breakdown_ratio <= self.ma5_breakdown_threshold:
                return True, f"跌破MA5({breakdown_ratio*100:.2f}%)"
            
            return False, f"未跌破MA5({breakdown_ratio*100:.2f}%)"
            
        except Exception as e:
            log.error(f"检查股票 {stock_code} MA5跌破时出错: {e}")
            return False, f"MA5检查出错: {e}"
    
    def _calculate_limit_up_price(self, stock_code: str, base_price: float) -> float:
        """
        计算涨停价格
        
        Args:
            stock_code: 股票代码
            base_price: 基准价格（前收盘价）
            
        Returns:
            涨停价格
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
            
        return base_price * (1 + limit_ratio)
    
    def _calculate_limit_down_price(self, stock_code: str, base_price: float) -> float:
        """
        计算跌停价格
        
        Args:
            stock_code: 股票代码
            base_price: 基准价格（前收盘价）
            
        Returns:
            跌停价格
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
            
        return base_price * (1 - limit_ratio)
    
    def get_sell_candidates(self, context, current_date) -> List[Dict]:
        """
        获取卖出候选股票列表
        
        Args:
            context: 聚宽上下文对象
            current_date: 当前日期
            
        Returns:
            卖出候选股票列表，格式: [{'code': str, 'reason': str, 'hold_days': int}, ...]
        """
        candidates = []
        
        # 遍历所有持仓
        for stock_code, position in context.portfolio.positions.items():
            if position.total_amount > 0:
                # 检查卖出条件
                should_sell, reason = self.check_sell_conditions(context, stock_code, current_date)
                
                if should_sell:
                    hold_days = self._calculate_hold_days(context, stock_code, current_date)
                    candidates.append({
                        'code': stock_code,
                        'reason': reason,
                        'hold_days': hold_days,
                        'position': position.total_amount,
                        'value': position.value
                    })
        
        return candidates
    
    def _get_ma5_value(self, stock_code, current_date):
        """
        获取MA5值
        
        Args:
            stock_code: 股票代码
            current_date: 当前日期
            
        Returns:
            float: MA5值，获取失败返回None
        """
        try:
            # 获取最近5天的价格数据
            price_data = get_price(stock_code, count=5, end_date=current_date, 
                                 fields=['close'])
            
            if len(price_data) < 5:
                return None
            
            # 计算MA5
            ma5 = price_data['close'].mean()
            return ma5
            
        except Exception as e:
            log.warning(f"获取股票 {stock_code} MA5值时出错: {e}")
            return None
    
    def calculate_ma5(self, stock_code: str, current_date, periods: int = 5) -> Optional[float]:
        """
        计算MA5均线值
        
        Args:
            stock_code: 股票代码
            current_date: 当前日期
            periods: 均线周期，默认5
            
        Returns:
            MA5值，计算失败返回None
        """
        try:
            # 获取足够的历史数据
            start_date = current_date - timedelta(days=periods * 2)
            
            price_data = get_price(
                stock_code,
                start_date=start_date,
                end_date=current_date,
                frequency='daily',
                fields=['close']
            )
            
            if len(price_data) < periods:
                return None
            
            # 计算MA5
            ma5_series = price_data['close'].rolling(window=periods).mean()
            return ma5_series.iloc[-1]
            
        except Exception as e:
            log.error(f"计算股票 {stock_code} MA5时出错: {e}")
            return None