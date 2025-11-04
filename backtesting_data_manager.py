"""
Backtesting.py 框架的数据管理器
负责从 xtquant 加载数据并转换为 Backtesting.py 兼容格式
完整迁移原始回测中的数据处理逻辑
"""

import os
import math
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import xtquant.xtdata as xtdata
from datetime import datetime, timedelta, time
import logging

# 导入原始回测中使用的模块
import first_limit_up_ma5_normal_scan as scan
import getAllStockCsv

class DataManager:
    """数据管理器，负责数据加载、处理和格式转换"""
    
    def __init__(self, config):
        """
        初始化数据管理器
        
        Args:
            config: 配置对象，包含回测参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 缓存已加载的数据
        self._data_cache = {}
        
        # 初始化查询工具
        self.query_tool = getAllStockCsv.StockQuery()
        
        # 扫描缓存相关
        self.cache_path = getattr(config, 'SCAN_CACHE_FILE', 'all_targets_cache_5.csv')
        self.scan_cache_df = None
        self.load_scan_cache()
        
        # 涨跌停比率
        self.LIMIT_RATE = getattr(config, 'LIMIT_UP_RATIO', 0.10)
    # --- 扫描缓存管理方法（从原始回测迁移） ---
    def load_scan_cache(self):
        """在回测开始时加载总的选股缓存文件到内存中。"""
        try:
            print(f"正在从 {self.cache_path} 加载选股缓存...")
            self.scan_cache_df = pd.read_csv(self.cache_path)
            # 确保date列是字符串类型，以便于比较
            self.scan_cache_df['date'] = self.scan_cache_df['date'].astype(str)
        except FileNotFoundError:
            print("缓存文件未找到，将创建一个新的。")
            self.scan_cache_df = pd.DataFrame(columns=['date', 'stock_code'])

    def get_target_stocks_from_cache(self, date_str):
        """
        获取目标股票列表。优先从内存缓存查找，否则调用扫描函数并更新缓存文件和内存。
        """
        # 1. 在内存中查找该日期是否已被缓存
        cached_stocks_df = self.scan_cache_df[self.scan_cache_df['date'] == date_str]

        # 检查'stock_code'列是否存在且有值，处理扫描过但无结果的情况 (NaN)
        if not cached_stocks_df.empty:
            print(f"从内存缓存中找到 {date_str} 的选股结果。")
            # 如果扫描过但没有结果，stock_code列可能是NaN，需要过滤掉
            return cached_stocks_df['stock_code'].dropna().tolist()

        # 2. 如果内存中没有，则执行扫描
        print(f"缓存未找到 {date_str} 的数据，执行实时扫描...")
        target_stocks, fourth_day_stocks = scan.get_target_stocks(target_date=date_str)

        # 3. 准备新数据以便存入缓存
        if target_stocks:
            new_data_df = pd.DataFrame({
                'date': [date_str] * len(target_stocks),
                'stock_code': list(target_stocks)
            })
        else:
            # 如果当天没有选出任何股票，也记录下来，date有值，stock_code为空(NaN)
            # 这样下次就不会重复扫描了
            new_data_df = pd.DataFrame([{'date': date_str, 'stock_code': None}])

        # 4. 将新数据追加到CSV文件
        # 'a'模式为追加, header只在文件不存在或为空时写入
        should_write_header = not os.path.exists(self.cache_path) or os.path.getsize(self.cache_path) == 0
        new_data_df.to_csv(self.cache_path, mode='a', header=should_write_header, index=False)

        # 5. 更新内存中的DataFrame
        self.scan_cache_df = pd.concat([self.scan_cache_df, new_data_df], ignore_index=True)

        # --- 数据加载方法（从原始回测迁移和增强） ---
    def get_daily_data(self, stock_code, start_date, end_date):
        """获取单个股票的日线历史数据。"""
        # 使用 get_local_data 获取日线数据通常更合适
        df = xtdata.get_local_data(
            field_list=['time', 'open', 'high', 'low', 'close', 'volume'],
            stock_list=[stock_code],
            period='1d',
            start_time=start_date,
            end_time=end_date,
            fill_data=True  # 确保对非交易日进行数据填充
        )
        if stock_code in df and not df[stock_code].empty:
            stock_df = df[stock_code].sort_index(ascending=True)
            # 关键：将索引转换为标准的datetime对象
            stock_df.index = pd.to_datetime(stock_df.index.astype(str), format='%Y%m%d')
            return stock_df
        return pd.DataFrame()

    def get_daily_data_with_metrics(self, stock_code, start_date, end_date):
        """获取日线数据并计算所需指标(MA5, 涨跌停价)。"""
        df = xtdata.get_local_data(
            field_list=['time', 'open', 'high', 'low', 'close', 'volume'],  # 需要pre_close来计算涨跌停
            stock_list=[stock_code],
            period='1d',
            start_time=start_date,
            end_time=end_date,
            fill_data=True
        )
        if stock_code not in df or df[stock_code].empty:
            return pd.DataFrame()

        stock_df = df[stock_code].sort_index(ascending=True)
        stock_df['pre_close'] = stock_df['close'].shift(1)
        stock_df['ma5'] = stock_df['close'].rolling(window=5, min_periods=1).mean()
        # 计算理论涨跌停价
        stock_df['limit_price'] = (stock_df['pre_close'] * (1 + self.LIMIT_RATE)).round(2)
        stock_df['down_limit_price'] = (stock_df['pre_close'] * (1 - self.LIMIT_RATE)).round(2)
        return stock_df

    def get_minute_data(self, stock_code, target_date):
        """获取单个股票在特定日期的1分钟K线数据。"""
        data = xtdata.get_local_data(
            field_list=['open', 'high', 'low', 'close'],
            stock_list=[stock_code],
            period='1m',
            start_time=target_date,
            end_time=target_date,
        )
        if stock_code in data and not data[stock_code].empty:
            df = data[stock_code]
            try:
                # 将数字索引转换为datetime对象
                df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d%H%M%S')
                return df
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def calculate_buy_target_price(self, stock_code, current_date_str):
        """
        基于前4天的MA5加上当天预测价格来计算目标买入价。
        """
        current_date = datetime.strptime(current_date_str, "%Y%m%d")
        # 获取当前日期之前的4个交易日的数据
        start_fetch_date = (current_date - timedelta(days=15)).strftime("%Y%m%d")
        end_fetch_date = (current_date - timedelta(days=1)).strftime("%Y%m%d")

        hist_df = self.get_daily_data(stock_code, start_fetch_date, end_fetch_date)

        if hist_df.empty or len(hist_df) < 4:
            print(f"  - [信息] {stock_code} 的历史数据不足，无法计算买入价。")
            return None

        # 取最后4天的数据
        last_4_days = hist_df.tail(4)
        if len(last_4_days) < 4:
            return None

        last_close = last_4_days.iloc[-1]['close']
        predicted_close = last_close * 1.04

        # 将历史收盘价与预测收盘价合并
        prices = last_4_days['close'].tolist()
        prices.append(predicted_close)

        # 计算5日均价
        target_price = round(sum(prices) / 5, 2)
        return target_price

    # --- Backtesting.py 框架兼容方法 ---
    def load_stock_data(self, stock_code: str, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        加载单只股票的历史数据并转换为Backtesting.py格式
            stock_code: 股票代码
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'
            
        Returns:
            DataFrame: 包含 OHLCV 数据的 DataFrame，索引为日期
        """
        try:
            # 使用配置中的日期范围
            if start_date is None:
                start_date = self.config.START_DATE
            if end_date is None:
                end_date = self.config.END_DATE
                
            # 检查缓存
            cache_key = f"{stock_code}_{start_date}_{end_date}"
            if cache_key in self._data_cache:
                return self._data_cache[cache_key].copy()
            
            # 从 xtquant 获取日线数据，使用与原始回测相同的API
            data = xtdata.get_local_data(
                field_list=['time', 'open', 'high', 'low', 'close', 'volume'],
                stock_list=[stock_code],
                period='1d',
                start_time=start_date,
                end_time=end_date,
                fill_data=True
            )
            
            if data is None or stock_code not in data:
                self.logger.warning(f"无法获取股票 {stock_code} 的数据")
                return None
                
            stock_data = data[stock_code]
            
            if stock_data.empty:
                self.logger.warning(f"股票 {stock_code} 数据为空")
                return None
            
            # 转换为 Backtesting.py 需要的格式
            df = self._convert_to_backtesting_format(stock_data)
            
            # 添加技术指标
            df = self._add_technical_indicators(df)
            
            # 缓存数据
            self._data_cache[cache_key] = df.copy()
            
            return df
            
        except Exception as e:
            self.logger.error(f"加载股票 {stock_code} 数据时出错: {e}")
            return None
    
    def _convert_to_backtesting_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        将 xtquant 数据格式转换为 Backtesting.py 格式
        
        Args:
            data: xtquant 原始数据
            
        Returns:
            DataFrame: Backtesting.py 格式的数据
        """
        # 重命名列以符合 Backtesting.py 的要求
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        df = data.rename(columns=column_mapping)
        
        # 确保索引是日期时间格式
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 确保数据类型正确
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        
        # 删除包含 NaN 的行
        df = df.dropna()
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标
        
        Args:
            df: 基础 OHLCV 数据
            
        Returns:
            DataFrame: 包含技术指标的数据
        """
        # 添加移动平均线
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # 添加前一日收盘价
        df['PrevClose'] = df['Close'].shift(1)
        
        # 计算涨停价格
        df['LimitUpPrice'] = df['PrevClose'] * (1 + self.config.LIMIT_UP_RATIO)
        
        # 判断是否涨停
        df['IsLimitUp'] = (df['Close'] >= df['LimitUpPrice'] * 0.995) & (df['Close'] <= df['LimitUpPrice'] * 1.005)
        
        # 计算涨跌幅
        df['PctChange'] = (df['Close'] - df['PrevClose']) / df['PrevClose'] * 100
        
        # 计算成交额（如果有成交量数据）
        if 'Volume' in df.columns:
            df['Amount'] = df['Close'] * df['Volume']
        
        return df
    
    def load_minute_data(self, stock_code: str, date: str) -> Optional[pd.DataFrame]:
        """
        加载指定日期的分钟数据
        
        Args:
            stock_code: 股票代码
            date: 日期，格式 'YYYYMMDD'
            
        Returns:
            DataFrame: 分钟级数据
        """
        try:
            # 获取分钟数据
            data = xt_data.get_market_data_ex(
                stock_list=[stock_code],
                period='1m',
                start_time=date,
                end_time=date,
                fill_data=True,
                dividend_type='none'
            )
            
            if data is None or stock_code not in data:
                return None
                
            minute_data = data[stock_code]
            
            if minute_data.empty:
                return None
            
            # 转换格式
            df = self._convert_to_backtesting_format(minute_data)
            
            return df
            
        except Exception as e:
            self.logger.error(f"加载股票 {stock_code} 在 {date} 的分钟数据时出错: {e}")
            return None
    
    def get_stock_pool(self) -> List[str]:
        """
        获取股票池
        
        Returns:
            List[str]: 股票代码列表
        """
        try:
            if hasattr(self.config, 'STOCK_POOL_FILE') and self.config.STOCK_POOL_FILE:
                # 从文件加载股票池
                df = pd.read_csv(self.config.STOCK_POOL_FILE)
                if 'stock_code' in df.columns:
                    return df['stock_code'].tolist()
                elif 'code' in df.columns:
                    return df['code'].tolist()
                else:
                    # 假设第一列是股票代码
                    return df.iloc[:, 0].tolist()
            else:
                # 默认股票池（可以根据需要修改）
                return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
                
        except Exception as e:
            self.logger.error(f"加载股票池时出错: {e}")
            return []
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据质量
        
        Args:
            df: 待验证的数据
            
        Returns:
            bool: 数据是否有效
        """
        if df is None or df.empty:
            return False
        
        # 检查必要的列
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            return False
        
        # 检查数据完整性
        if df[required_columns].isnull().any().any():
            return False
        
        # 检查价格逻辑
        invalid_prices = (df['High'] < df['Low']) | (df['High'] < df['Open']) | (df['High'] < df['Close']) | \
                        (df['Low'] > df['Open']) | (df['Low'] > df['Close'])
        
        if invalid_prices.any():
            self.logger.warning("发现价格逻辑错误的数据")
            return False
        
        return True
    
    def clear_cache(self):
        """清空数据缓存"""
        self._data_cache.clear()