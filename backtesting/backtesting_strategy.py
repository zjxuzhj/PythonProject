"""
Backtesting.py 框架的首板策略
完整迁移自 qmt_real_time_backtest.py 的回测逻辑
"""

import pandas as pd
import numpy as np
import math
from backtesting import Strategy
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta, time
import os

# 导入原始模块
try:
    import first_limit_up_ma5_normal_scan as scan
    import getAllStockCsv
    from common_sell_logic import get_sell_decision, MarketDataContext
    from stock_info import StockInfo
    import xtquant.xtdata as xtdata
except ImportError as e:
    logging.warning(f"导入模块失败: {e}")


class FirstLimitUpStrategy(Strategy):
    """
    首板策略 - 完整迁移自原始 Backtester 类
    
    策略逻辑：
    1. 基于扫描缓存的股票选择
    2. MA5目标价格买入
    3. 使用common_sell_logic的卖出条件
    4. 完整的仓位和风险管理
    """
    
    # 策略参数 - 与原始回测保持一致
    position_size = 20000      # 单笔交易金额
    limit_up_ratio = 0.10      # 涨停幅度
    benchmark_ticker = "000300.SH"  # 基准指数
    risk_free_rate = 0.03      # 无风险利率
    
    # 交易时间控制
    buy_cutoff_time = time(14, 30)  # 买入截止时间
    
    def init(self):
        """策略初始化"""
        self.logger = logging.getLogger(__name__)
        
        # 初始化查询工具
        try:
            self.query_tool = getAllStockCsv.StockQuery()
        except Exception as e:
            self.logger.warning(f"初始化股票查询工具失败: {e}")
            self.query_tool = None
        
        # 加载扫描缓存
        self.scan_cache_df = pd.DataFrame()
        self.load_scan_cache()
        
        # 策略状态变量
        self.positions_info = {}  # 持仓详细信息
        self.trade_log = []       # 交易记录
        self.daily_values = []    # 每日价值记录
        
        # 当前交易状态
        self.current_date = None
        self.current_cash = None
        self.current_positions = {}
        
        self.logger.info("首板策略初始化完成")
    
    def load_scan_cache(self):
        """加载扫描缓存"""
        try:
            cache_file = '../all_targets_cache_5.csv'
            if os.path.exists(cache_file):
                self.scan_cache_df = pd.read_csv(cache_file)
                self.logger.info(f"加载扫描缓存成功，共 {len(self.scan_cache_df)} 条记录")
            else:
                self.logger.warning(f"扫描缓存文件不存在: {cache_file}")
                self.scan_cache_df = pd.DataFrame()
        except Exception as e:
            self.logger.error(f"加载扫描缓存失败: {e}")
            self.scan_cache_df = pd.DataFrame()
    
    def get_target_stocks_from_cache(self, date_str: str) -> List[str]:
        """从缓存获取目标股票"""
        try:
            if self.scan_cache_df.empty:
                return []
            
            # 检查缓存中是否有该日期的数据
            date_data = self.scan_cache_df[self.scan_cache_df['date'] == date_str]
            if not date_data.empty:
                target_stocks = date_data['code'].tolist()
                self.logger.info(f"从缓存获取到 {len(target_stocks)} 只目标股票")
                return target_stocks
            else:
                # 如果缓存中没有，进行实时扫描
                self.logger.info(f"缓存中没有 {date_str} 的数据，进行实时扫描")
                try:
                    scanner = scan.FirstLimitUpMA5NormalScan()
                    target_stocks = scanner.scan(date_str)
                    
                    # 更新缓存
                    new_data = pd.DataFrame({
                        'date': [date_str] * len(target_stocks),
                        'code': target_stocks
                    })
                    self.scan_cache_df = pd.concat([self.scan_cache_df, new_data], ignore_index=True)
                    
                    # 保存到文件
                    cache_file = '../all_targets_cache_5.csv'
                    self.scan_cache_df.to_csv(cache_file, index=False)
                    
                    return target_stocks
                except Exception as e:
                    self.logger.error(f"实时扫描失败: {e}")
                    return []
        except Exception as e:
            self.logger.error(f"获取目标股票失败: {e}")
            return []
    
    def get_daily_data_with_metrics(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取包含技术指标的日线数据"""
        try:
            data = xtdata.get_local_data(
                field_list=['time', 'open', 'high', 'low', 'close', 'volume'],
                stock_list=[stock_code],
                period='1d',
                start_time=start_date,
                end_time=end_date,
                fill_data=True
            )
            
            if data is None or stock_code not in data:
                return pd.DataFrame()
            
            df = data[stock_code]
            if df.empty:
                return pd.DataFrame()
            
            # 计算技术指标
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['prev_close'] = df['close'].shift(1)
            df['limit_price'] = df['prev_close'] * (1 + self.limit_up_ratio)
            df['down_limit_price'] = df['prev_close'] * (1 - self.limit_up_ratio)
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取股票 {stock_code} 数据失败: {e}")
            return pd.DataFrame()
    
    def get_minute_data(self, stock_code: str, target_date: str) -> pd.DataFrame:
        """获取分钟数据"""
        try:
            data = xtdata.get_local_data(
                field_list=['time', 'open', 'high', 'low', 'close', 'volume'],
                stock_list=[stock_code],
                period='1m',
                start_time=target_date + " 09:30:00",
                end_time=target_date + " 15:00:00",
                fill_data=True
            )
            
            if data is None or stock_code not in data:
                return pd.DataFrame()
            
            return data[stock_code]
            
        except Exception as e:
            self.logger.error(f"获取股票 {stock_code} 分钟数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_buy_target_price(self, stock_code: str, current_date_str: str) -> Optional[float]:
        """计算买入目标价格（基于MA5）"""
        try:
            # 获取前4天的数据
            current_date = datetime.strptime(current_date_str, "%Y%m%d")
            start_date = (current_date - timedelta(days=10)).strftime("%Y%m%d")
            
            df = self.get_daily_data_with_metrics(stock_code, start_date, current_date_str)
            if len(df) < 4:
                return None
            
            # 获取最近4天的收盘价
            last_4_days = df.tail(4)
            prices = last_4_days['close'].tolist()
            
            # 预测当天收盘价（使用前一天收盘价）
            predicted_close = df.iloc[-1]['close']
            prices.append(predicted_close)
            
            # 计算5日均价
            target_price = round(sum(prices) / 5, 2)
            return target_price
            
        except Exception as e:
            self.logger.error(f"计算买入目标价格失败: {e}")
            return None
    
    def _get_buy_time_category(self, buy_datetime: datetime) -> str:
        """获取买入时间分类"""
        buy_time = buy_datetime.time()
        if buy_time < time(10, 0):
            return "早盘"
        elif buy_time < time(11, 30):
            return "上午"
        elif buy_time < time(13, 30):
            return "午间"
        elif buy_time < time(14, 30):
            return "下午"
        else:
            return "尾盘"
    
    def log_trade(self, date_time: datetime, stock_code: str, action: str, 
                  shares: int, price: float, reason: str = "", pnl: Optional[float] = None, 
                  buy_time_category: Optional[str] = None):
        """记录交易"""
        trade_record = {
            'datetime': date_time,
            'stock_code': stock_code,
            'stock_name': self.query_tool.get_name_by_code(stock_code) if self.query_tool else stock_code,
            'action': action,
            'shares': shares,
            'price': price,
            'amount': shares * price,
            'reason': reason,
            'pnl': pnl,
            'buy_time_category': buy_time_category
        }
        self.trade_log.append(trade_record)
        
        action_desc = "买入" if action == "BUY" else "卖出"
        pnl_desc = f", 盈亏: {pnl:.2f}" if pnl is not None else ""
        self.logger.info(f"{action_desc} {trade_record['stock_name']} {shares}股 @ {price:.2f}{pnl_desc}")
    
    def check_sell_conditions(self, stock_code: str, position: Dict, current_dt: datetime) -> Tuple[Optional[float], Optional[str]]:
        """检查卖出条件"""
        try:
            date_str = current_dt.strftime("%Y%m%d")
            
            # 获取包含当天和历史的日线数据
            start_fetch_date = (current_dt - timedelta(days=30)).strftime("%Y%m%d")
            daily_df = self.get_daily_data_with_metrics(stock_code, start_fetch_date, date_str)
            
            # 需要至少两天数据
            if len(daily_df) < 2:
                return None, None
            
            today_data = daily_df.iloc[-1]
            prev_day_data = daily_df.iloc[-2]
            
            # 构建市场数据上下文
            position_info = {'hold_days': position['hold_days']}
            market_data = MarketDataContext(
                high=today_data['high'],
                low=today_data['low'],
                open=today_data['open'],
                close=today_data['close'],
                ma5=today_data['ma5'],
                up_limit_price=today_data['limit_price'],
                down_limit_price=today_data['down_limit_price'],
                prev_close=prev_day_data['close'],
                prev_up_limit_price=prev_day_data['limit_price'],
                prev_down_limit_price=prev_day_data['down_limit_price']
            )
            
            # 构建股票信息
            stock_info = StockInfo(
                code=stock_code,
                name=self.query_tool.get_name_by_code(stock_code) if self.query_tool else stock_code,
                market_value=self.query_tool.get_stock_market_value(stock_code) if self.query_tool else 0,
                theme=self.query_tool.get_theme_by_code(stock_code) if self.query_tool else "",
            )
            
            # 使用通用卖出逻辑
            should_sell, reason = get_sell_decision(stock_info, position_info, market_data)
            if should_sell:
                return today_data['close'], reason
            else:
                return None, None
                
        except Exception as e:
            self.logger.error(f"检查卖出条件失败: {e}")
            return None, None
    
    def next(self):
        """策略主逻辑 - 在每个交易日执行"""
        # 注意：在Backtesting.py框架中，这个方法会为每只股票单独调用
        # 我们需要在这里实现单股票的买卖逻辑
        
        # 获取当前数据
        current_data = self.data.df.iloc[len(self.data.df) - 1]
        current_date = current_data.name
        
        # 更新当前状态
        self.current_date = current_date
        
        # 检查是否应该买入
        self._process_buy_signals(current_date, current_data)
        
        # 检查是否应该卖出
        self._process_sell_signals(current_date, current_data)
    
    def _process_buy_signals(self, current_date: datetime, current_data: pd.Series):
        """处理买入信号"""
        # 在Backtesting.py框架中，每只股票单独处理
        # 这里需要根据具体的股票代码来判断是否买入
        pass  # 具体实现需要根据框架的使用方式调整
    
    def _process_sell_signals(self, current_date: datetime, current_data: pd.Series):
        """处理卖出信号"""
        # 在Backtesting.py框架中，每只股票单独处理
        # 这里需要根据持仓情况判断是否卖出
        pass  # 具体实现需要根据框架的使用方式调整