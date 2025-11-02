# -*- coding: utf-8 -*-
"""
聚宽策略 - 买入时机模块
基于MA5回踩的买入条件判断和执行逻辑
"""

import jqdata
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional


class BuyTimingManager:
    """买入时机管理器 - 负责买入条件判断和执行"""
    
    def __init__(self):
        # 配置参数
        self.valid_buy_days = [2, 4]  # 首板后第2天或第4天可买入
        self.ma5_tolerance = 0.02  # MA5附近的容差范围（±2%）
        self.max_position_ratio = 0.10  # 单只股票最大仓位比例（10%）
        self.slippage_ratio = 0.02  # 滑点容差（2%）
        
    def check_buy_conditions(self, context, stock_code: str, stock_pool_manager, current_date) -> Tuple[bool, str]:
        """
        检查买入条件
        
        Args:
            context: 聚宽上下文对象
            stock_code: 股票代码
            stock_pool_manager: 股票池管理器
            current_date: 当前日期
            
        Returns:
            (是否可买入, 原因说明)
        """
        try:
            # 1. 检查是否在跟踪池中
            day_count = stock_pool_manager.get_stock_day_count(stock_code)
            if day_count == -1:
                return False, "不在跟踪池中"
            
            # 2. 检查是否为有效买入日（第2天或第4天）
            if day_count not in self.valid_buy_days:
                return False, f"非买入日(当前第{day_count}天)"
            
            # 3. 检查当日是否跌停
            if self._is_limit_down(stock_code, current_date):
                return False, "当日跌停"
            
            # 4. 检查是否回踩至MA5附近
            ma5_check, ma5_reason = self._check_ma5_pullback(stock_code, current_date)
            if not ma5_check:
                return False, ma5_reason
            
            # 5. 检查仓位限制
            if self._check_position_limit(context, stock_code):
                return False, "仓位已满"
            
            return True, f"满足买入条件(第{day_count}天,MA5回踩)"
            
        except Exception as e:
            log.error(f"检查股票 {stock_code} 买入条件时出错: {e}")
            return False, f"检查出错: {e}"
    
    def execute_buy_order(self, context, stock_code: str, current_date) -> bool:
        """
        执行买入订单
        
        Args:
            context: 聚宽上下文对象
            stock_code: 股票代码
            current_date: 当前日期
            
        Returns:
            是否成功下单
        """
        try:
            # 计算买入金额（总资金的10%）
            total_value = context.portfolio.total_value
            buy_amount = total_value * self.max_position_ratio
            
            # 获取当前价格
            current_data = get_current_data()
            if stock_code not in current_data:
                log.warning(f"无法获取股票 {stock_code} 的当前价格")
                return False
            
            current_price = current_data[stock_code].last_price
            
            # 计算滑点后的价格
            slippage_price = current_price * (1 + self.slippage_ratio)
            
            # 下市价单（使用order_target_value按金额买入）
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
        """
        判断是否跌停
        
        Args:
            stock_code: 股票代码
            current_date: 当前日期
            
        Returns:
            bool: 是否跌停
        """
        try:
            # 获取当日价格数据
            price_data = get_price(stock_code, count=1, end_date=current_date, 
                                 fields=['close', 'pre_close'])
            
            if price_data.empty:
                return False
            
            close_price = price_data.iloc[-1]['close']
            pre_close = price_data.iloc[-1]['pre_close']
            
            # 计算跌停价格
            limit_down_price = self._calculate_limit_down_price(stock_code, pre_close)
            
            # 判断是否跌停（允许0.01的误差）
            return abs(close_price - limit_down_price) <= 0.01
            
        except Exception as e:
            log.warning(f"判断股票 {stock_code} 是否跌停时出错: {e}")
            return False
    
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
    
    def _check_ma5_pullback(self, stock_code, current_date):
        """
        检查是否回踩至MA5附近（±2%）
        
        Args:
            stock_code: 股票代码
            current_date: 当前日期
            
        Returns:
            bool: 是否符合MA5回踩条件
        """
        try:
            # 获取最近6天的价格数据（包含当天）
            price_data = get_price(stock_code, count=6, end_date=current_date, 
                                 fields=['close'])
            
            if len(price_data) < 6:
                return False
            
            # 计算MA5（使用前5天的数据）
            ma5 = price_data.iloc[:-1]['close'].mean()
            
            # 获取当前价格
            current_price = price_data.iloc[-1]['close']
            
            # 检查是否在MA5的±2%范围内
            tolerance = self.ma5_tolerance
            lower_bound = ma5 * (1 - tolerance)
            upper_bound = ma5 * (1 + tolerance)
            
            is_pullback = lower_bound <= current_price <= upper_bound
            
            log.debug(f"{stock_code} MA5回踩检查: 当前价格={current_price:.2f}, "
                     f"MA5={ma5:.2f}, 范围=[{lower_bound:.2f}, {upper_bound:.2f}], "
                     f"符合条件={is_pullback}")
            
            return is_pullback
            
        except Exception as e:
            log.warning(f"检查股票 {stock_code} MA5回踩时出错: {e}")
            return False
    
    def _check_position_limit(self, context, stock_code: str) -> bool:
        """
        检查仓位限制
        
        Args:
            context: 聚宽上下文对象
            stock_code: 股票代码
            
        Returns:
            是否超过仓位限制
        """
        try:
            # 检查是否已持有该股票
            if stock_code in context.portfolio.positions:
                current_position = context.portfolio.positions[stock_code]
                if current_position.total_amount > 0:
                    return True  # 已持有，不再买入
            
            # 检查总持仓数量
            total_positions = len([pos for pos in context.portfolio.positions.values() 
                                 if pos.total_amount > 0])
            
            # 限制最大持仓数量（例如10只）
            max_positions = 10
            if total_positions >= max_positions:
                return True
            
            return False
            
        except Exception as e:
            log.error(f"检查仓位限制时出错: {e}")
            return True  # 出错时保守处理
    
    def get_buy_candidates(self, context, stock_pool_manager, current_date) -> List[Dict]:
        """
        获取买入候选股票列表
        
        Args:
            context: 聚宽上下文对象
            stock_pool_manager: 股票池管理器
            current_date: 当前日期
            
        Returns:
            买入候选股票列表，格式: [{'code': str, 'reason': str, 'day_count': int}, ...]
        """
        candidates = []
        
        # 获取跟踪池中的所有股票
        tracking_stocks = stock_pool_manager.tracking_pool
        
        for stock_code, stock_info in tracking_stocks.items():
            # 检查买入条件
            can_buy, reason = self.check_buy_conditions(
                context, stock_code, stock_pool_manager, current_date
            )
            
            if can_buy:
                candidates.append({
                    'code': stock_code,
                    'name': stock_info['name'],
                    'reason': reason,
                    'day_count': stock_info['day_count'],
                    'first_limit_price': stock_info['first_limit_price']
                })
        
        # 按首板后天数排序（优先买入第2天的）
        candidates.sort(key=lambda x: x['day_count'])
        
        return candidates
    
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