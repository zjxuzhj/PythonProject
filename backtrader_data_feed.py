"""
Backtrader 数据适配器
将 xtquant 数据格式转换为 Backtrader 兼容的数据源
"""

import pandas as pd
import backtrader as bt
import xtquant.xtdata as xtdata
from datetime import datetime, timedelta


class XtQuantDataFeed(bt.feeds.PandasData):
    """
    xtquant 数据源适配器
    继承自 Backtrader 的 PandasData，用于处理 xtquant 格式的数据
    """
    
    # 定义数据列映射
    params = (
        ('datetime', None),  # 使用索引作为时间
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),  # xtquant 没有持仓量数据
        # 添加自定义技术指标列
        ('ma5', 'ma5'),
        ('ma10', 'ma10'),
        ('ma20', 'ma20'),
        ('ma30', 'ma30'),
        ('ma60', 'ma60'),
        ('ma120', 'ma120'),
        ('limit_price', 'limit_price'),
        ('down_limit_price', 'down_limit_price'),
        ('is_limit', 'is_limit'),
        ('pre_close', 'pre_close'),
    )
    
    # 定义数据线，用于在策略中访问自定义指标
    lines = ('ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'ma120', 
             'limit_price', 'down_limit_price', 'is_limit', 'pre_close')


class DataManager:
    """
    数据管理器，负责从 xtquant 获取数据并转换为 Backtrader 格式
    """
    
    def __init__(self, limit_rate=0.10):
        self.limit_rate = limit_rate
        
    def get_stock_data(self, stock_code, start_date, end_date, period='1d'):
        """
        获取股票数据并计算技术指标
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            period: 数据周期，默认日线 '1d'
            
        Returns:
            pandas.DataFrame: 包含OHLCV和技术指标的数据
        """
        try:
            # 从 xtquant 获取原始数据
            df = xtdata.get_local_data(
                field_list=['time', 'open', 'high', 'low', 'close', 'volume'],
                stock_list=[stock_code],
                period=period,
                start_time=start_date,
                end_time=end_date,
                fill_data=True
            )
            
            if stock_code not in df or df[stock_code].empty:
                return pd.DataFrame()
                
            stock_df = df[stock_code].sort_index(ascending=True)
            
            # 计算技术指标
            stock_df = self._calculate_indicators(stock_df)
            
            return stock_df
            
        except Exception as e:
            print(f"获取股票 {stock_code} 数据失败: {e}")
            return pd.DataFrame()
    
    def _calculate_indicators(self, df):
        """
        计算技术指标
        
        Args:
            df: 原始OHLCV数据
            
        Returns:
            pandas.DataFrame: 包含技术指标的数据
        """
        # 计算前收盘价
        df['pre_close'] = df['close'].shift(1)
        
        # 计算移动平均线
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['ma30'] = df['close'].rolling(window=30, min_periods=1).mean()
        df['ma60'] = df['close'].rolling(window=60, min_periods=1).mean()
        df['ma120'] = df['close'].rolling(window=120, min_periods=1).mean()
        
        # 计算涨跌停价格
        df['limit_price'] = (df['pre_close'] * (1 + self.limit_rate)).round(2)
        df['down_limit_price'] = (df['pre_close'] * (1 - self.limit_rate)).round(2)
        
        # 判断是否涨停
        df['is_limit'] = (abs(df['close'] - df['limit_price']) < 0.01) & (df['close'] > 0)
        
        return df
    
    def create_data_feed(self, stock_code, start_date, end_date, period='1d'):
        """
        创建 Backtrader 数据源
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            period: 数据周期
            
        Returns:
            XtQuantDataFeed: Backtrader 数据源对象
        """
        df = self.get_stock_data(stock_code, start_date, end_date, period)
        
        if df.empty:
            return None
            
        # 确保索引是 datetime 类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # 创建数据源
        data_feed = XtQuantDataFeed(
            dataname=df,
            name=stock_code,
            fromdate=datetime.strptime(start_date, '%Y%m%d'),
            todate=datetime.strptime(end_date, '%Y%m%d')
        )
        
        return data_feed
    
    def get_minute_data(self, stock_code, target_date):
        """
        获取分钟级数据
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期 (YYYYMMDD)
            
        Returns:
            pandas.DataFrame: 分钟级数据，索引为 datetime 对象
        """
        try:
            df = xtdata.get_local_data(
                field_list=['time', 'open', 'high', 'low', 'close', 'volume'],
                stock_list=[stock_code],
                period='1m',
                start_time=target_date,
                end_time=target_date,
                fill_data=True
            )
            
            if stock_code not in df or df[stock_code].empty:
                return pd.DataFrame()
                
            stock_df = df[stock_code].sort_index(ascending=True)
            
            # 检查原始索引类型
            original_index_type = type(stock_df.index)
            
            # 将数字索引转换为datetime对象
            try:
                if isinstance(stock_df.index, pd.DatetimeIndex):
                    # 如果已经是 DatetimeIndex，直接返回
                    return stock_df
                else:
                    # 转换为 datetime 索引
                    stock_df.index = pd.to_datetime(stock_df.index.astype(str), format='%Y%m%d%H%M%S')
                    
                # 验证转换结果
                if not isinstance(stock_df.index, pd.DatetimeIndex):
                    print(f"警告: 索引转换后仍不是 DatetimeIndex，类型为: {type(stock_df.index)}")
                    return pd.DataFrame()
                    
                # 验证第一个元素是否有 .time() 方法
                if len(stock_df) > 0:
                    first_element = stock_df.index[0]
                    if not hasattr(first_element, 'time'):
                        print(f"警告: 索引元素没有 .time() 方法，类型为: {type(first_element)}")
                        return pd.DataFrame()
                        
            except Exception as e:
                print(f"转换时间索引失败 (股票: {stock_code}, 日期: {target_date}): {e}")
                print(f"原始索引类型: {original_index_type}")
                print(f"原始索引前5个元素: {stock_df.index[:5].tolist() if len(stock_df) > 0 else '无数据'}")
                return pd.DataFrame()
                
            return stock_df
            
        except Exception as e:
            print(f"获取股票 {stock_code} 分钟数据失败: {e}")
            return pd.DataFrame()


class MultiStockDataManager:
    """
    多股票数据管理器，用于管理多只股票的数据源
    """
    
    def __init__(self, limit_rate=0.10):
        self.data_manager = DataManager(limit_rate)
        self.data_feeds = {}
        
    def add_stock_data(self, stock_code, start_date, end_date, period='1d'):
        """
        添加股票数据源
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 数据周期
        """
        data_feed = self.data_manager.create_data_feed(
            stock_code, start_date, end_date, period
        )
        
        if data_feed is not None:
            self.data_feeds[stock_code] = data_feed
            
    def get_data_feed(self, stock_code, start_date=None, end_date=None, period='1d'):
        """
        获取指定股票的数据源，如果不存在则创建
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 数据周期
            
        Returns:
            XtQuantDataFeed: 数据源对象
        """
        # 如果已存在数据源，直接返回
        if stock_code in self.data_feeds:
            return self.data_feeds[stock_code]
        
        # 如果不存在且提供了日期参数，则创建新的数据源
        if start_date and end_date:
            self.add_stock_data(stock_code, start_date, end_date, period)
            return self.data_feeds.get(stock_code)
        
        return None
    
    def get_all_data_feeds(self):
        """
        获取所有数据源
        
        Returns:
            dict: 所有数据源字典
        """
        return self.data_feeds
    
    def clear_data_feeds(self):
        """清空所有数据源"""
        self.data_feeds.clear()


# 工具函数
def convert_date_format(date_str):
    """
    转换日期格式
    
    Args:
        date_str: 日期字符串 (YYYYMMDD)
        
    Returns:
        datetime: datetime 对象
    """
    return datetime.strptime(date_str, '%Y%m%d')


def format_stock_code(code):
    """
    格式化股票代码为 xtquant 标准格式
    
    Args:
        code: 股票代码
        
    Returns:
        str: 标准格式的股票代码
    """
    if '.' not in code:
        if code.startswith('6'):
            return f"{code}.SH"
        elif code.startswith(('0', '3')):
            return f"{code}.SZ"
        elif code.startswith('8') or code.startswith('4'):
            return f"{code}.BJ"
    return code