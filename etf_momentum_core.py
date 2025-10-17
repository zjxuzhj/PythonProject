"""
ETF动量轮动策略核心模块
统一的ETF池配置和动量计算方法，供回测和实盘策略共享使用
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import xtquant.xtdata as xtdata


class ETFMomentumCore:
    """ETF动量计算核心类"""
    
    # 统一的ETF池配置
    ETF_POOL = [
        # 境外
        "513100.SH",  # 纳指ETF
        "159509.SZ",  # 纳指科技ETF
        "513520.SH",  # 日经ETF
        "513030.SH",  # 德国ETF
        # 商品
        "518880.SH",  # 黄金ETF 2013.7 714亿
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
        "159851.SZ",  # 金融科技 2021.3 123亿
        "159637.SZ",  # 新能源车
        "516160.SH",  # 新能源 2021.2 61亿 中证新能
        "159550.SZ",  # 互联网ETF
        "512710.SH",  # 军工ETF
        "159692.SZ",  # 证券
        "512480.SH",  # 半导体 2019.6 230亿
    ]
    
    # 统一的ETF名称映射
    ETF_NAMES = {
        "513100.SH": "纳指ETF",
        "159509.SZ": "纳指科技ETF",
        "513520.SH": "日经ETF",
        "513030.SH": "德国ETF",
        "518880.SH": "黄金ETF",
        "159980.SZ": "有色ETF",
        "159985.SZ": "豆粕ETF",
        "501018.SH": "南方原油",
        "511090.SH": "30年国债ETF",
        "513130.SH": "恒生科技",
        "513690.SH": "港股红利",
        "510180.SH": "上证180",
        "159915.SZ": "创业板ETF",
        "510410.SH": "资源",
        "515650.SH": "消费50",
        "512290.SH": "生物医药",
        "588120.SH": "科创100",
        "515070.SH": "人工智能ETF",
        "159851.SZ": "金融科技",
        "159637.SZ": "新能源车",
        "516160.SH": "新能源",
        "159550.SZ": "互联网ETF",
        "512710.SH": "军工ETF",
        "159692.SZ": "证券",
        "512480.SH": "半导体",
    }
    
    def __init__(self, momentum_days: int = 25):
        """
        初始化ETF动量核心类
        
        Args:
            momentum_days: 动量计算天数，默认25天
        """
        self.momentum_days = momentum_days
        self.etf_pool = self.ETF_POOL.copy()
        self.etf_names = self.ETF_NAMES.copy()
    
    def get_historical_data_for_backtest(self, stock_code: str, days: int, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取回测用的历史数据（指定结束日期）
        
        Args:
            stock_code: 股票代码
            days: 需要的天数
            end_date: 结束日期，格式'YYYYMMDD'
            
        Returns:
            DataFrame或None
        """
        try:
            # 计算开始日期（向前推更多天数以确保有足够的交易日）
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            start_dt = end_dt - timedelta(days=days + 30)  # 多取30天以确保有足够交易日
            start_date = start_dt.strftime('%Y%m%d')
            
            # 获取本地数据
            data = xtdata.get_local_data(
                field_list=['open', 'high', 'low', 'close', 'volume'],
                stock_list=[stock_code],
                period='1d',
                start_time=start_date,
                end_time=end_date,
            )
            
            if stock_code not in data or data[stock_code].empty:
                return None
            
            df = data[stock_code]
            # 确保有足够的数据
            if len(df) >= days:
                return df.tail(days)
            else:
                return None
                
        except Exception as e:
            print(f"获取 {stock_code} 历史数据异常: {str(e)}")
            return None
    
    def get_historical_data_for_live(self, stock_code: str, days: int) -> Optional[pd.DataFrame]:
        """
        获取实盘用的历史数据（到昨天为止）
        
        Args:
            stock_code: 股票代码
            days: 需要的天数
            
        Returns:
            DataFrame或None
        """
        try:
            # 获取历史数据
            yesterday = datetime.now() - timedelta(days=1)
            end_date = yesterday.strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days + 40)).strftime('%Y%m%d')
            
            data = xtdata.get_local_data(
                field_list=['close'],
                stock_list=[stock_code],
                period='1d',
                start_time=start_date,
                end_time=end_date,
            )

            if stock_code in data and len(data[stock_code]) > 0:
                df = data[stock_code]
                # 确保有足够的数据
                if len(df) >= days:
                    return df.tail(days)

            return None

        except Exception as e:
            print(f"获取 {stock_code} 历史数据异常: {str(e)}")
            return None
    
    def get_price_at_date(self, stock_code: str, target_date: str) -> Optional[float]:
        """
        获取指定日期的价格（用于回测）
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期，格式'YYYYMMDD'
            
        Returns:
            价格或None
        """
        try:
            # 首先尝试获取目标日期的数据
            data = xtdata.get_local_data(
                field_list=['close'],
                stock_list=[stock_code],
                period='1d',
                start_time=target_date,
                end_time=target_date,
            )
            
            if stock_code in data and not data[stock_code].empty:
                return float(data[stock_code]['close'].iloc[0])
            else:
                # 如果当天没有数据，尝试获取最近的交易日数据
                end_dt = datetime.strptime(target_date, '%Y%m%d')
                start_dt = end_dt - timedelta(days=7)  # 向前查找7天
                start_date = start_dt.strftime('%Y%m%d')
                
                data = xtdata.get_local_data(
                    field_list=['close'],
                    stock_list=[stock_code],
                    period='1d',
                    start_time=start_date,
                    end_time=target_date,
                )
                
                if stock_code in data and not data[stock_code].empty:
                    df = data[stock_code]
                    df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d')
                    target_dt = datetime.strptime(target_date, '%Y%m%d')
                    df = df[df.index <= target_dt]
                    if not df.empty:
                        return float(df['close'].iloc[-1])
                
                return None
                
        except Exception as e:
            print(f"获取 {stock_code} 在 {target_date} 的价格异常: {str(e)}")
            return None
    
    def get_current_price(self, stock_code: str) -> float:
        """
        获取当前价格（用于实盘）
        
        Args:
            stock_code: 股票代码
            
        Returns:
            当前价格
        """
        try:
            tick = xtdata.get_full_tick([stock_code])
            if stock_code in tick and tick[stock_code]:
                return tick[stock_code].get('lastPrice', 0)
            return 0
        except Exception as e:
            print(f"获取 {stock_code} 当前价格异常: {str(e)}")
            return 0
    
    def calculate_momentum_score(self, stock_code: str, prices: np.ndarray) -> Optional[Dict]:
        """
        统一的动量得分计算方法
        
        Args:
            stock_code: 股票代码
            prices: 价格序列（包含历史价格和当前/目标价格）
            
        Returns:
            包含动量得分信息的字典或None
        """
        try:
            if len(prices) < self.momentum_days + 1:
                return None
            
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
                'current_price': prices[-1],
                'data_points': len(prices) - 1
            }
            
        except Exception as e:
            print(f"计算 {stock_code} 动量得分异常: {str(e)}")
            return None
    
    def calculate_momentum_score_for_backtest(self, stock_code: str, target_date: str) -> Optional[Dict]:
        """
        计算回测用的动量得分
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期，格式'YYYYMMDD'
            
        Returns:
            包含动量得分信息的字典或None
        """
        try:
            # 获取历史数据（不包含目标日期）
            target_dt = datetime.strptime(target_date, '%Y%m%d')
            prev_date = (target_dt - timedelta(days=1)).strftime('%Y%m%d')
            
            df = self.get_historical_data_for_backtest(stock_code, self.momentum_days, prev_date)
            if df is None or len(df) < self.momentum_days:
                return None
            
            # 获取目标日期的价格
            current_price = self.get_price_at_date(stock_code, target_date)
            if current_price is None or current_price <= 0:
                return None
            
            # 构建价格序列（历史价格 + 目标日期价格）
            prices = np.append(df['close'].values, current_price)
            
            return self.calculate_momentum_score(stock_code, prices)
            
        except Exception as e:
            print(f"计算 {stock_code} 回测动量得分异常: {str(e)}")
            return None
    
    def calculate_momentum_score_for_live(self, stock_code: str) -> Optional[Dict]:
        """
        计算实盘用的动量得分
        
        Args:
            stock_code: 股票代码
            
        Returns:
            包含动量得分信息的字典或None
        """
        try:
            # 获取历史数据
            df = self.get_historical_data_for_live(stock_code, self.momentum_days)
            if df is None or len(df) < self.momentum_days:
                return None

            # 获取当前价格
            current_price = self.get_current_price(stock_code)
            if current_price <= 0:
                return None

            # 构建价格序列（历史价格 + 当前价格）
            prices = np.append(df['close'].values, current_price)
            
            return self.calculate_momentum_score(stock_code, prices)

        except Exception as e:
            print(f"计算 {stock_code} 实盘动量得分异常: {str(e)}")
            return None
    
    def get_etf_scores_for_backtest(self, target_date: str) -> List[Dict]:
        """
        获取回测用的所有ETF得分
        
        Args:
            target_date: 目标日期，格式'YYYYMMDD'
            
        Returns:
            ETF得分列表，按得分降序排列
        """
        etf_scores = []
        
        for etf_code in self.etf_pool:
            score_data = self.calculate_momentum_score_for_backtest(etf_code, target_date)
            if score_data:
                etf_scores.append(score_data)
        
        # 按得分排序
        etf_scores.sort(key=lambda x: x['score'], reverse=True)
        return etf_scores
    
    def get_etf_scores_for_live(self) -> List[Dict]:
        """
        获取实盘用的所有ETF得分
        
        Returns:
            ETF得分列表，按得分降序排列
        """
        etf_scores = []
        
        for etf_code in self.etf_pool:
            score_data = self.calculate_momentum_score_for_live(etf_code)
            if score_data:
                etf_scores.append(score_data)
        
        # 按得分排序
        etf_scores.sort(key=lambda x: x['score'], reverse=True)
        return etf_scores
    
    def get_top_etf(self, etf_scores: List[Dict]) -> Optional[Dict]:
        """
        获取得分最高的ETF
        
        Args:
            etf_scores: ETF得分列表
            
        Returns:
            得分最高的ETF信息或None
        """
        valid_scores = [score for score in etf_scores if score['score'] > 0]
        if valid_scores:
            return valid_scores[0]
        return None
    
    def print_etf_scores(self, etf_scores: List[Dict], date_str: str = None):
        """
        打印ETF得分表格
        
        Args:
            etf_scores: ETF得分列表
            date_str: 日期字符串，用于显示
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n{'=' * 75}")
        print(f"ETF分值排序结果 ({date_str})")
        print(f"{'=' * 75}")
        print(f"{'排名':<4} {'代码':<12} {'名称':<12} {'分值':<8} {'年化收益':<10} {'R2':<8} {'价格':<8} {'数据点'}")
        print("-" * 75)

        for rank, data in enumerate(etf_scores, 1):
            etf_name = self.etf_names.get(data['stock_code'], '未知')
            print(f"{rank:<4} {data['stock_code']:<12} {etf_name:<12} "
                  f"{data['score']:<8.4f} {data['annualized_returns']:<10.2%} "
                  f"{data['r2']:<8.4f} {data['current_price']:<8.2f} {data['data_points']}")
        print(f"{'=' * 75}\n")