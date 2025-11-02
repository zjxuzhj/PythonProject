# -*- coding: utf-8 -*-
"""
聚宽策略 - 股票池选择模块
基于首板涨停策略的股票筛选和动态跟踪池管理
"""

import jqdata
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set


class StockPoolManager:
    """股票池管理器 - 负责首板股票的筛选和跟踪"""
    
    def __init__(self):
        # 动态跟踪池：存储首板股票的跟踪信息
        # 格式: {stock_code: {'first_limit_date': date, 'first_limit_price': float, 'day_count': int}}
        self.tracking_pool = {}
        
        # 配置参数
        self.min_market_cap = 50 * 100000000  # 50亿（聚宽单位：元）
        self.max_market_cap = 200 * 100000000  # 200亿
        self.min_list_days = 60  # 上市满60个交易日
        self.track_days = 5  # 跟踪5个交易日
        
    def get_target_stocks(self, context, current_date):
        """
        获取目标股票列表（聚宽版本）
        
        Args:
            context: 聚宽上下文对象
            current_date: 当前日期
            
        Returns:
            list: 目标股票列表
        """
        try:
            # 1. 获取所有A股股票
            all_stocks = list(get_all_securities(['stock'], current_date).index)
            
            # 2. 筛选首板涨停股票
            first_limit_stocks = []
            
            for stock_code in all_stocks:
                try:
                    # 检查基本条件
                    if not self._check_basic_conditions(stock_code, current_date):
                        continue
                    
                    # 检查是否为首次涨停
                    if self._is_first_limit_up(stock_code, current_date):
                        # 获取股票信息
                        stock_info = self._get_stock_info(stock_code, current_date)
                        if stock_info:
                            first_limit_stocks.append({
                                'code': stock_code,
                                'name': stock_info['name'],
                                'first_limit_date': current_date,
                                'first_limit_price': stock_info['price'],
                                'day_count': 1
                            })
                
                except Exception as e:
                    log.warning(f"处理股票 {stock_code} 时出错: {e}")
                    continue
            
            # 3. 更新跟踪池
            self._update_tracking_pool(first_limit_stocks, current_date)
            
            # 4. 返回当前跟踪池中的所有股票
            return list(self.tracking_pool.values())
            
        except Exception as e:
            log.error(f"获取目标股票时出错: {e}")
            return []
    
    def _update_tracking_pool(self, current_date):
        """更新跟踪池中所有股票的day_count"""
        for stock_code in list(self.tracking_pool.keys()):
            stock_info = self.tracking_pool[stock_code]
            # 计算交易日差
            days_diff = self._get_trading_days_diff(
                stock_info['first_limit_date'], 
                current_date
            )
            self.tracking_pool[stock_code]['day_count'] = days_diff
    
    def _get_first_limit_stocks(self, context, current_date) -> List[Dict]:
        """
        获取当日首板涨停股票
        
        Returns:
            首板股票信息列表，格式: [{'code': str, 'price': float, 'name': str}, ...]
        """
        # 获取所有A股股票
        all_stocks = self._get_all_stocks(context, current_date)
        
        first_limit_stocks = []
        
        for stock in all_stocks:
            # 检查基本条件
            if not self._check_basic_conditions(stock, current_date):
                continue
                
            # 检查是否为当日首板涨停
            if self._is_first_limit_up(stock, current_date):
                # 获取股票基本信息
                stock_info = self._get_stock_info(stock, current_date)
                if stock_info:
                    first_limit_stocks.append(stock_info)
        
        return first_limit_stocks
    
    def _get_all_stocks(self, context, current_date) -> List[str]:
        """获取所有A股股票代码"""
        # 获取所有股票
        stocks = list(get_all_securities(['stock'], current_date).index)
        
        # 过滤掉ST股票和科创板、创业板的特殊情况
        filtered_stocks = []
        for stock in stocks:
            # 获取股票信息
            security_info = get_security_info(stock)
            
            # 排除ST股票
            if 'ST' in security_info.display_name or '*ST' in security_info.display_name:
                continue
                
            # 排除退市股票
            if security_info.end_date < current_date:
                continue
                
            filtered_stocks.append(stock)
            
        return filtered_stocks
    
    def _check_basic_conditions(self, stock_code, current_date):
        """
        检查股票基本条件
        
        Args:
            stock_code: 股票代码
            current_date: 当前日期
            
        Returns:
            bool: 是否符合基本条件
        """
        try:
            # 1. 检查是否为ST股票
            if self._is_st_stock(stock_code):
                return False
            
            # 2. 检查上市时间（需要满60个交易日）
            stock_info = get_security_info(stock_code)
            if not stock_info:
                return False
            
            list_days = self._get_trading_days_between(stock_info.start_date, current_date)
            if list_days < self.min_list_days:
                return False
            
            # 3. 检查流通市值
            market_cap = self._get_circulating_market_cap(stock_code, current_date)
            if not (self.min_market_cap <= market_cap <= self.max_market_cap):
                return False
            
            return True
            
        except Exception as e:
            log.warning(f"检查股票 {stock_code} 基本条件时出错: {e}")
            return False
    
    def _is_first_limit_up(self, stock_code: str, current_date) -> bool:
        """
        判断是否为首板涨停
        
        Args:
            stock_code: 股票代码
            current_date: 当前日期
            
        Returns:
            是否为首板涨停
        """
        try:
            # 获取最近5个交易日的数据
            end_date = current_date
            start_date = current_date - timedelta(days=10)  # 多取几天确保有足够的交易日
            
            # 获取价格数据
            price_data = get_price(
                stock_code, 
                start_date=start_date, 
                end_date=end_date, 
                frequency='daily',
                fields=['close', 'high', 'low', 'open']
            )
            
            if len(price_data) < 2:
                return False
            
            # 获取当日和前一日数据
            today_data = price_data.iloc[-1]
            yesterday_data = price_data.iloc[-2]
            
            # 计算涨停价（根据板块不同有不同的涨停幅度）
            limit_up_price = self._calculate_limit_up_price(stock_code, yesterday_data['close'])
            
            # 判断当日是否涨停
            is_limit_today = abs(today_data['close'] - limit_up_price) < 0.01
            
            if not is_limit_today:
                return False
            
            # 判断是否为首板（前一日未涨停）
            yesterday_limit_price = self._calculate_limit_up_price(
                stock_code, 
                price_data.iloc[-3]['close'] if len(price_data) >= 3 else yesterday_data['close']
            )
            
            is_limit_yesterday = abs(yesterday_data['close'] - yesterday_limit_price) < 0.01
            
            # 首板条件：今日涨停且昨日未涨停
            return is_limit_today and not is_limit_yesterday
            
        except Exception as e:
            log.error(f"判断股票 {stock_code} 是否首板涨停时出错: {e}")
            return False
    
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
    
    def _get_stock_info(self, stock_code: str, current_date) -> Dict:
        """
        获取股票基本信息
        
        Args:
            stock_code: 股票代码
            current_date: 当前日期
            
        Returns:
            股票信息字典
        """
        try:
            # 获取股票名称
            security_info = get_security_info(stock_code)
            
            # 获取当日收盘价
            price_data = get_price(
                stock_code,
                start_date=current_date,
                end_date=current_date,
                frequency='daily',
                fields=['close']
            )
            
            if price_data.empty:
                return None
                
            return {
                'code': stock_code,
                'name': security_info.display_name,
                'price': price_data.iloc[0]['close']
            }
            
        except Exception as e:
            log.error(f"获取股票 {stock_code} 信息时出错: {e}")
            return None
    
    def _is_st_stock(self, stock_code):
        """
        判断是否为ST股票
        """
        try:
            security_info = get_security_info(stock_code)
            return 'ST' in security_info.display_name or '*ST' in security_info.display_name
        except:
            return True
    
    def _get_trading_days_between(self, start_date, end_date):
        """
        获取两个日期之间的交易日数
        """
        try:
            trading_days = get_trade_days(start_date=start_date, end_date=end_date)
            return len(trading_days) - 1  # 不包含起始日
        except:
            return 0
    
    def _get_circulating_market_cap(self, stock_code, current_date):
        """
        获取流通市值
        """
        try:
            # 使用聚宽财务数据接口
            q = query(valuation.circulating_market_cap).filter(valuation.code == stock_code)
            df = get_fundamentals(q, date=current_date)
            
            if df.empty:
                return 0
            
            # 返回流通市值（单位：元）
            return df.iloc[0]['circulating_market_cap'] * 100000000
            
        except Exception as e:
            log.warning(f"获取股票 {stock_code} 流通市值时出错: {e}")
            return 0
    
    def _add_to_tracking_pool(self, first_limit_stocks: List[Dict], current_date):
        """
        将首板股票加入跟踪池
        
        Args:
            first_limit_stocks: 首板股票列表
            current_date: 当前日期
        """
        for stock_info in first_limit_stocks:
            stock_code = stock_info['code']
            
            # 如果股票不在跟踪池中，则加入
            if stock_code not in self.tracking_pool:
                self.tracking_pool[stock_code] = {
                    'first_limit_date': current_date,
                    'first_limit_price': stock_info['price'],
                    'day_count': 0,
                    'name': stock_info['name']
                }
                log.info(f"新增首板股票到跟踪池: {stock_code} {stock_info['name']}")
    
    def _clean_expired_stocks(self):
        """清理超过跟踪期的股票"""
        expired_stocks = []
        
        for stock_code, stock_info in self.tracking_pool.items():
            if stock_info['day_count'] > self.track_days:
                expired_stocks.append(stock_code)
        
        for stock_code in expired_stocks:
            stock_info = self.tracking_pool.pop(stock_code)
            log.info(f"移除超期股票: {stock_code} {stock_info['name']} (跟踪{stock_info['day_count']}天)")
    
    def _get_trading_days_diff(self, start_date, end_date) -> int:
        """
        计算两个日期之间的交易日差
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易日差
        """
        try:
            trading_days = get_trade_days(start_date=start_date, end_date=end_date)
            return len(trading_days) - 1  # 减1是因为不包含起始日
        except:
            return 0
    
    def get_stock_day_count(self, stock_code: str) -> int:
        """
        获取股票的首板后天数
        
        Args:
            stock_code: 股票代码
            
        Returns:
            首板后天数，如果不在跟踪池中返回-1
        """
        if stock_code in self.tracking_pool:
            return self.tracking_pool[stock_code]['day_count']
        return -1
    
    def get_stock_first_limit_price(self, stock_code: str) -> float:
        """
        获取股票的首板价格
        
        Args:
            stock_code: 股票代码
            
        Returns:
            首板价格，如果不在跟踪池中返回0
        """
        if stock_code in self.tracking_pool:
            return self.tracking_pool[stock_code]['first_limit_price']
        return 0.0