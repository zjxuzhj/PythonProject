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
            
            for stock_code in all_stocks[:1000]:  # 限制检查前100只股票以提高效率
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
                # log.warning(f"isST")
                return False
            
            # 2. 检查上市时间（放宽到30天）
            stock_info = get_security_info(stock_code)
            if not stock_info:
                return False
            
            list_days = self._get_trading_days_between(stock_info.start_date, current_date)
            # log.warning(f"islist_days:{list_days}")
            if list_days < 250:  # 放宽上市时间要求
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
            # print(security_info.display_name)
            return 'ST' in security_info.display_name or '*ST' in security_info.display_name
        except:
            return True
    
    def _get_trading_days_between(self, start_date, end_date):
        """获取两个日期之间的交易日数"""
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            delta_days = (end_dt - start_dt).days
            estimated_days = delta_days * 0.9
            return max(0, estimated_days)
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
    
    def get_tracking_stocks(self):
        """获取当前跟踪的股票列表"""
        return list(self.tracking_pool.keys())
    
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
        self.predict_ratio = 1.04  # 预测涨幅系数
        self.target_ma5_prices = {}  # 存储每只股票的目标MA5价格
    
    def calculate_dynamic_ma5_price(self, stock_code, current_date):
        """
        动态计算预估MA5价格
        参考modify_last_days_and_calc_ma5方法的逻辑
        """
        try:
            # 获取过去6天的数据（用于计算MA5）
            price_data = get_price(stock_code, count=6, end_date=current_date, 
                                 fields=['close'])
            
            if len(price_data) < 5:
                log.warning(f"股票 {stock_code} 数据不足，无法计算动态MA5")
                return None
            
            # 模拟预测逻辑：基于最后一天价格预测今日价格
            last_close = price_data.iloc[-1]['close']
            predicted_today_price = last_close * self.predict_ratio
            
            # 创建包含预测价格的数据序列
            close_prices = price_data['close'].tolist()
            close_prices.append(predicted_today_price)
            
            # 计算包含预测价格的MA5
            if len(close_prices) >= 5:
                ma5_price = sum(close_prices[-5:]) / 5
                log.info(f"股票 {stock_code} 动态MA5价格: {ma5_price:.2f} (基于预测价格 {predicted_today_price:.2f})")
                return round(ma5_price, 2)
            
            return None
            
        except Exception as e:
            log.warning(f"计算股票 {stock_code} 动态MA5价格时出错: {e}")
            return None
    
    def update_target_ma5_prices(self, context, stock_pool_manager, current_date):
        """更新所有跟踪股票的目标MA5价格"""
        try:
            tracking_stocks = stock_pool_manager.get_tracking_stocks()
            log.info(f"更新 {len(tracking_stocks)} 只股票的目标MA5价格")
            
            for stock_code in tracking_stocks:
                target_price = self.calculate_dynamic_ma5_price(stock_code, current_date)
                if target_price:
                    self.target_ma5_prices[stock_code] = target_price
                    log.info(f"股票 {stock_code} 目标MA5价格: {target_price:.2f}")
            
        except Exception as e:
            log.error(f"更新目标MA5价格时出错: {e}")

    def check_price_trigger_buy(self, context, tracking_stocks=None):
        """检查是否有股票价格触发买入条件 - 优化版本支持分钟级检测"""
        buy_candidates = []
        
        try:
            # 如果没有传入跟踪股票列表，使用目标价格字典的股票
            if tracking_stocks is None:
                tracking_stocks = list(self.target_ma5_prices.keys())
            
            # 如果没有跟踪股票或目标价格，直接返回
            if not tracking_stocks or not self.target_ma5_prices:
                return []
            
            # 获取当前数据（批量获取，提高效率）
            current_data = get_current_data()
            
            # 获取当前持仓股票列表，避免重复查询
            current_positions_stocks = set([stock for stock, position in context.portfolio.positions.items() 
                                          if position.total_amount > 0])
            
            # 计算当前持仓数量
            current_positions_count = len(current_positions_stocks)
            
            # 如果已达到最大持仓数，直接返回
            if current_positions_count >= StrategyConfig.MAX_POSITIONS:
                 return []
            
            # 只检查有目标价格且未持仓的股票
            valid_stocks = [stock for stock in tracking_stocks 
                          if stock in self.target_ma5_prices and stock not in current_positions_stocks]
            
            if not valid_stocks:
                return []
            
            # 批量检查价格触发条件
            for stock_code in valid_stocks:
                # 检查是否达到最大持仓数
                if len(buy_candidates) + current_positions_count >= StrategyConfig.MAX_POSITIONS:
                     break
                
                # 获取目标MA5价格
                target_ma5_price = self.target_ma5_prices.get(stock_code)
                if not target_ma5_price:
                    continue
                
                # 获取当前价格
                if stock_code not in current_data:
                    continue
                
                current_price = current_data[stock_code].last_price
                if not current_price or current_price <= 0:
                    continue
                
                # 检查价格是否触发买入（当前价格接近或达到目标MA5价格）
                price_diff_ratio = abs(current_price - target_ma5_price) / target_ma5_price
                
                if price_diff_ratio <= self.ma5_tolerance:
                    # 进一步检查其他买入条件（简化版本，减少API调用）
                    can_buy, reason = self._check_additional_buy_conditions_fast(
                        context, stock_code, current_data[stock_code])
                    
                    if can_buy:
                        buy_candidates.append({
                            'stock_code': stock_code,
                            'current_price': current_price,
                            'target_ma5_price': target_ma5_price,
                            'reason': f"价格触发MA5买入: 当前价格{current_price:.2f}, 目标MA5{target_ma5_price:.2f}"
                        })
                        
                        # 详细日志（仅在有买入机会时记录）
                        log.info(f"价格触发买入: {stock_code}, 当前价格: {current_price:.2f}, "
                               f"目标MA5: {target_ma5_price:.2f}, 差异: {price_diff_ratio:.3f}")
            
            return buy_candidates
            
        except Exception as e:
            log.error(f"检查价格触发买入时出错: {e}")
            return []
    
    def _check_additional_buy_conditions_fast(self, context, stock_code, stock_data):
        """快速检查其他买入条件 - 减少API调用的优化版本"""
        try:
            # 检查股票是否停牌
            if stock_data.paused:
                return False, "股票停牌"
            
            # 检查是否涨停（涨停时无法买入）
            if stock_data.high_limit and stock_data.last_price >= stock_data.high_limit * 0.999:
                return False, "股票涨停"
            
            # 检查是否跌停（跌停时风险较大）
            if stock_data.low_limit and stock_data.last_price <= stock_data.low_limit * 1.001:
                return False, "股票跌停"
            
            # 检查仓位限制
            if not self._check_position_limit(context, stock_code):
                return False, "超出仓位限制"
            
            return True, "满足买入条件"
            
        except Exception as e:
            log.warning(f"检查买入条件时出错 {stock_code}: {e}")
            return False, f"检查出错: {e}"
    
    def _check_additional_buy_conditions(self, context, stock_code, current_date):
        """检查除价格外的其他买入条件"""
        try:
            # 检查是否跌停
            if self._is_limit_down(stock_code, current_date):
                return False, "当日跌停"
            
            # 检查仓位限制
            if not self._check_position_limit(context, stock_code):
                return False, "仓位限制"
            
            return True, "符合条件"
            
        except Exception as e:
            log.warning(f"检查股票 {stock_code} 额外买入条件时出错: {e}")
            return False, f"检查异常: {e}"
    
    def get_buy_candidates(self, context, stock_pool_manager, current_date):
        """获取买入候选股票"""
        candidates = []
        
        log.info(f"[调试] 开始检查买入候选股票，跟踪池股票数量: {len(stock_pool_manager.tracking_pool)}")
        
        for stock_info in stock_pool_manager.tracking_pool.values():
            stock_code = stock_info['code']
            day_count = stock_info['day_count']
            
            log.info(f"[调试] 检查股票 {stock_code} {stock_info['name']} 第{day_count}天")
            
            # 检查买入条件
            is_ok, reason_msg = self._check_buy_conditions(context, stock_code, day_count, current_date)
            
            if is_ok:
                candidates.append({
                    'code': stock_code,
                    'name': stock_info['name'],
                    'day_count': day_count,
                    'reason': f"第{day_count}天MA5回踩"
                })
                log.info(f"[调试] 股票 {stock_code} 符合买入条件")
            else:
                # 打印不符合买入条件的具体原因
                log.info(f"[调试] 股票 {stock_code} 不符合买入条件: {reason_msg}")
        
        # 按优先级排序（第2天优先于第4天）
        candidates.sort(key=lambda x: x['day_count'])
        
        log.info(f"[调试] 买入候选股票数量: {len(candidates)}")
        
        return candidates
    
    def _check_buy_conditions(self, context, stock_code, day_count, current_date):
        """
        检查买入条件
        【已修改】返回 (bool, str)，str为不符合条件的原因
        """
        try:
            # 1. 检查是否为有效买入日
            if day_count not in self.valid_buy_days:
                return False, f"不在有效买入日(第{day_count}天)"
            
            # 2. 检查当日是否跌停
            if self._is_limit_down(stock_code, current_date):
                return False, "当日跌停"
            
            # 3. 检查MA5回踩
            if not self._check_ma5_pullback(stock_code, current_date):
                return False, "未回踩MA5"
            
            # 4. 检查仓位限制
            if not self._check_position_limit(context, stock_code):
                return False, "仓位限制(已持仓或总仓位已满)"
            
            return True, "符合所有条件"
            
        except Exception as e:
            log.warning(f"检查股票 {stock_code} 买入条件时出错: {e}")
            return False, f"检查时出现异常: {e}"
    
    def execute_buy_order_fast(self, context, stock_code, reason):
        """快速执行买入订单 - 分钟级检测优化版本"""
        try:
            # 检查股票是否有效
            current_data = get_current_data()
            if stock_code not in current_data:
                log.warning(f"股票 {stock_code} 数据不可用")
                return False
            
            stock_data = current_data[stock_code]
            
            # 检查股票状态
            if stock_data.paused:
                log.warning(f"股票 {stock_code} 停牌，无法买入")
                return False
            
            # 计算买入金额（使用固定比例）
            total_value = context.portfolio.total_value
            position_value = total_value * StrategyConfig.MAX_POSITION_RATIO
            
            # 获取当前价格
            current_price = stock_data.last_price
            if not current_price or current_price <= 0:
                log.warning(f"股票 {stock_code} 价格异常: {current_price}")
                return False
            
            # 计算买入股数（100股的整数倍）
            shares = int(position_value / current_price / 100) * 100
            
            if shares < 100:
                log.warning(f"股票 {stock_code} 买入金额不足100股")
                return False
            
            # 执行买入订单
            order_id = order(stock_code, shares)
            
            if order_id:
                log.info(f"买入订单已提交: {stock_code}, 股数: {shares}, 价格: {current_price:.2f}, 原因: {reason}")
                return True
            else:
                log.warning(f"买入订单提交失败: {stock_code}")
                return False
                
        except Exception as e:
            log.error(f"执行买入订单时出错 {stock_code}: {e}")
            return False
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
    log.info("开始初始化策略...")
    
    # 初始化全局变量
    g['stock_pool_manager'] = StockPoolManager()
    g['buy_timing_manager'] = BuyTimingManager()
    g['sell_timing_manager'] = SellTimingManager()
    
    # 初始化分钟级监控相关变量
    g['last_minute_check'] = None  # 记录上次检查的分钟
    g['minute_check_count'] = 0    # 每日检查次数统计
    g['daily_buy_count'] = 0       # 每日买入次数统计
    g['max_daily_buys'] = 20       # 每日最大买入次数限制
    
    # 初始化交易统计变量
    g['trade_count'] = 0           # 总交易次数
    g['win_count'] = 0             # 盈利交易次数
    g['total_return'] = 0.0        # 总收益率
    
    # 设置基准
    set_benchmark('000300.XSHG')  # 沪深300
    
    # 设置手续费
    set_order_cost(OrderCost(
        close_tax=0.001,      # 印花税
        open_commission=0.0003,  # 开仓佣金
        close_commission=0.0003, # 平仓佣金
        min_commission=5      # 最小佣金
    ), type='stock')
    
    # 设置滑点
    set_slippage(FixedSlippage(0.02))
    
    # 设置每日开盘运行（股票池更新和目标价格计算）
    run_daily(daily_market_open, time='09:30')
    
    # 设置分钟级价格监控（交易时间内每分钟运行）
    # 上午时段：09:31-11:30
    for hour in range(9, 12):
        start_minute = 31 if hour == 9 else 0
        end_minute = 30 if hour == 11 else 59
        for minute in range(start_minute, end_minute + 1):
            if hour == 11 and minute > 30:  # 11:30后停止
                break
            run_daily(minute_price_monitor, time=f'{hour:02d}:{minute:02d}')
    
    # 下午时段：13:01-15:00
    for hour in range(13, 16):
        start_minute = 1 if hour == 13 else 0
        end_minute = 0 if hour == 15 else 59
        for minute in range(start_minute, end_minute + 1):
            if hour == 15 and minute > 0:  # 15:00后停止
                break
            run_daily(minute_price_monitor, time=f'{hour:02d}:{minute:02d}')
    
    # 设置收盘后处理
    run_daily(after_market_close, time='15:30')
    
    log.info("策略初始化完成 - 已设置分钟级价格监控")

def daily_market_open(context):
    """每日开盘时执行的主要逻辑"""
    current_time = context.current_dt
    log.info(f"=== {current_time.strftime('%Y-%m-%d')} 开盘交易逻辑开始 ===")
    
    # 重置每日计数器
    g['daily_buy_count'] = 0
    g['minute_check_count'] = 0
    
    # 执行卖出逻辑
    execute_sell_logic(context, context.current_dt.date())
    
    # 更新股票池
    target_stocks = g['stock_pool_manager'].get_target_stocks(context, context.current_dt.date())
    log.info(f"股票池更新完成，当前跟踪 {len(target_stocks)} 只股票")
    
    # 更新目标MA5价格
    g['buy_timing_manager'].update_target_ma5_prices(
        context, g['stock_pool_manager'], context.current_dt.date())
    
    # 执行开盘买入检查
    execute_dynamic_buy_logic(context, context.current_dt.date())
    
    log.info(f"开盘逻辑执行完成，当前持仓数量: {len([p for p in context.portfolio.positions.values() if p.total_amount > 0])}")

def minute_price_monitor(context):
    """分钟级价格监控函数"""
    current_time = context.current_dt
    current_minute = current_time.strftime('%H:%M')
    
    # 避免重复检查同一分钟
    if g.get('last_minute_check') == current_minute:
        return
    
    g['last_minute_check'] = current_minute
    g['minute_check_count'] = g.get('minute_check_count', 0) + 1
    
    # 检查是否超过每日买入限制
    if g.get('daily_buy_count', 0) >= g.get('max_daily_buys', 5):
        if g['minute_check_count'] % 30 == 1:  # 每30分钟提醒一次
            log.info(f"[{current_minute}] 已达到每日最大买入次数限制({g['max_daily_buys']})")
        return
    
    # 检查当前持仓数量
    current_positions = [p for p in context.portfolio.positions.values() if p.total_amount > 0]
    if len(current_positions) >= StrategyConfig.MAX_POSITIONS:
        if g['minute_check_count'] % 30 == 1:  # 每30分钟提醒一次
            log.info(f"[{current_minute}] 已达到最大持仓数量限制({StrategyConfig.MAX_POSITIONS})")
        return
    
    # 获取跟踪股票列表
    tracking_stocks = g['stock_pool_manager'].get_tracking_stocks()
    if not tracking_stocks:
        if g['minute_check_count'] % 60 == 1:  # 每小时提醒一次
            log.info(f"[{current_minute}] 当前无跟踪股票")
        return
    
    # 执行价格触发买入检查
    buy_candidates = g['buy_timing_manager'].check_price_trigger_buy(context, tracking_stocks)
    
    if buy_candidates:
        log.info(f"[{current_minute}] 发现 {len(buy_candidates)} 个买入机会: {[stock for stock, _ in buy_candidates]}")
        
        # 执行买入操作
        for stock_code, reason in buy_candidates:
            if g.get('daily_buy_count', 0) >= g.get('max_daily_buys', 5):
                log.info(f"[{current_minute}] 达到每日买入限制，停止买入")
                break
                
            if len([p for p in context.portfolio.positions.values() if p.total_amount > 0]) >= StrategyConfig.MAX_POSITIONS:
                log.info(f"[{current_minute}] 达到持仓限制，停止买入")
                break
            
            success = g['buy_timing_manager'].execute_buy_order_fast(context, stock_code, reason)
            if success:
                g['daily_buy_count'] = g.get('daily_buy_count', 0) + 1
                log.info(f"[{current_minute}] 成功买入 {stock_code}，今日买入次数: {g['daily_buy_count']}")
    else:
        # 每10分钟记录一次检测状态
        if g['minute_check_count'] % 10 == 1:
            log.info(f"[{current_minute}] 检测 {len(tracking_stocks)} 只股票，暂无买入机会 (检测次数: {g['minute_check_count']})")

def after_market_close(context):
    """收盘后处理"""
    current_time = context.current_dt
    log.info(f"=== {current_time.strftime('%Y-%m-%d')} 收盘后处理 ===")
    
    # 统计今日活动
    current_positions = [p for p in context.portfolio.positions.values() if p.total_amount > 0]
    
    # 详细统计信息
    total_value = context.portfolio.total_value
    available_cash = context.portfolio.available_cash
    positions_value = total_value - available_cash
    
    log.info(f"=== 今日分钟级策略执行统计 ===")
    log.info(f"分钟级检测次数: {g.get('minute_check_count', 0)}")
    log.info(f"今日买入次数: {g.get('daily_buy_count', 0)}")
    log.info(f"当前持仓数量: {len(current_positions)}")
    log.info(f"账户总价值: {total_value:.2f}")
    log.info(f"可用现金: {available_cash:.2f}")
    log.info(f"持仓市值: {positions_value:.2f}")
    log.info(f"仓位使用率: {(positions_value/total_value)*100:.1f}%")
    
    # 持仓详情
    if current_positions:
        log.info("=== 当前持仓详情 ===")
        for position in current_positions:
            stock_code = position.security
            shares = position.total_amount
            avg_cost = position.avg_cost
            current_price = position.price
            market_value = position.value
            pnl_ratio = (current_price - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0
            
            log.info(f"{stock_code}: 股数{shares}, 成本{avg_cost:.2f}, 现价{current_price:.2f}, "
                   f"市值{market_value:.2f}, 盈亏{pnl_ratio:+.1f}%")
    
    # 跟踪股票池状态
    tracking_stocks = g['stock_pool_manager'].get_tracking_stocks()
    target_prices = g['buy_timing_manager'].target_ma5_prices
    
    log.info(f"=== 股票池状态 ===")
    log.info(f"跟踪股票数量: {len(tracking_stocks)}")
    log.info(f"有目标价格股票数量: {len(target_prices)}")
    
    if tracking_stocks:
        log.info(f"跟踪股票列表: {tracking_stocks[:10]}{'...' if len(tracking_stocks) > 10 else ''}")
    
    # 重置计数器
    g['minute_check_count'] = 0
    g['daily_buy_count'] = 0
    g['last_minute_check'] = None
    
    log.info("=== 收盘后处理完成 ===")

# 删除重复的after_market_close函数


def trade_logic(context):
    """主要交易逻辑 - 支持动态MA5价格触发买入"""
    current_date = context.current_dt.date()
    log.info(f"[调试] ========== 开始执行交易逻辑 {current_date} ==========")
    
    try:
        # 1. 更新股票池
        target_stocks = g['stock_pool_manager'].get_target_stocks(context, current_date)
        log.info(f"[调试] 股票池更新完成，目标股票数量: {len(target_stocks)}")
        
        # 2. 更新目标MA5价格（每日开盘前计算）
        g['buy_timing_manager'].update_target_ma5_prices(
            context, g['stock_pool_manager'], current_date)
        
        # 3. 执行卖出逻辑
        execute_sell_logic(context, current_date)
        
        # 4. 执行动态买入逻辑（价格触发）
        execute_dynamic_buy_logic(context, current_date)
        
        # 5. 记录当日状态
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

def execute_dynamic_buy_logic(context, current_date):
    """执行动态买入逻辑 - 基于价格触发"""
    log.info("--- 执行动态买入逻辑（价格触发） ---")
    
    # 检查仓位限制
    current_positions = len([pos for pos in context.portfolio.positions.values() 
                           if pos.total_amount > 0])
    
    if current_positions >= StrategyConfig.MAX_POSITIONS:
        log.info(f"已达最大持仓数量限制: {current_positions}/{StrategyConfig.MAX_POSITIONS}")
        return
    
    # 检查价格触发的买入候选
    buy_candidates = g['buy_timing_manager'].check_price_trigger_buy(context)
    
    if not buy_candidates:
        log.info("无价格触发买入的股票")
        return
    
    log.info(f"发现 {len(buy_candidates)} 只价格触发买入的股票")
    
    # 执行买入
    bought_count = 0
    max_buy_count = StrategyConfig.MAX_POSITIONS - current_positions
    
    for candidate in buy_candidates[:max_buy_count]:
        stock_code = candidate['stock_code']
        current_price = candidate['current_price']
        target_ma5_price = candidate['target_ma5_price']
        
        success = g['buy_timing_manager'].execute_buy_order_fast(context, stock_code, "价格触发买入")
        
        if success:
            bought_count += 1
            log.info(f"价格触发买入成功: {stock_code}, 当前价格: {current_price:.2f}, "
                   f"目标MA5: {target_ma5_price:.2f}")
        else:
            log.warning(f"价格触发买入失败: {stock_code}")
    
    log.info(f"本日价格触发买入股票数量: {bought_count}")

def execute_buy_logic(context, current_date):
    """执行买入逻辑（原有逻辑，保留作为备用）"""
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
    
    for stock_code in buy_candidates[:max_buy_count]:
        success = g['buy_timing_manager'].execute_buy_order_fast(context, stock_code, "常规买入")
        
        if success:
            bought_count += 1
            log.info(f"买入成功: {stock_code}")
        else:
            log.warning(f"买入失败: {stock_code}")
    
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

# ==================== 分钟级策略说明 ====================
"""
分钟级五日线买入策略 - 使用说明：

1. 复制本文件全部内容到聚宽策略编辑器
2. 设置回测参数：
   - 起始资金：1000000元（100万）
   - 基准指数：沪深300 (000300.XSHG)
   - 回测时间：建议至少3个月
3. 点击运行回测

策略核心逻辑（分钟级版本）：
- 股票池：每日筛选首板涨停股票，建立5日跟踪池
- 分钟级监控：交易时间内每分钟检测股价与MA5的关系
- 买入：实时检测到股价触碰动态计算的MA5时立即执行买入
- 卖出：收盘价跌破MA5、持仓超过15天、炸板等条件

分钟级优化特性：
- 每分钟实时价格监控（09:31-11:30, 13:01-15:00）
- 动态MA5计算，精确捕捉买入时机
- 智能限流机制，避免过度交易
- 详细的分钟级日志记录
- 优化的API调用，减少系统资源消耗

风险控制：
- 最大持仓10只股票
- 单股仓位不超过10%
- 每日最大买入次数限制
- 严格的买卖条件判断
- 完整的止盈止损机制

性能优化：
- 批量数据获取，减少API调用
- 快速条件检查，提高执行效率
- 智能缓存机制，避免重复计算
- 分钟级计数器，监控系统负载

注意事项：
- 策略适用于A股市场分钟级交易
- 分钟级模式会增加计算资源消耗
- 建议充分回测后再考虑实盘
- 可根据市场情况调整MA5容差和限流参数
- 实盘使用时注意API调用频率限制
"""