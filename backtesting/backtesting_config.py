"""
Backtesting.py 框架配置文件
包含所有回测相关的参数配置
"""

from datetime import datetime, timedelta
import os

class BacktestingConfig:
    """Backtesting.py 框架配置类"""
    
    def __init__(self):
        """初始化配置参数"""
        
        # ==================== 基础配置 ====================
        # 初始资金
        self.INITIAL_CASH = 1000000  # 100万
        
        # 手续费设置
        self.COMMISSION = 0.0003  # 万分之3
        
        # 滑点设置
        self.SLIPPAGE = 0.001  # 0.1%
        
        # ==================== 时间配置 ====================
        # 回测时间范围
        self.START_DATE = '20240101'
        self.END_DATE = '20241231'
        
        # ==================== 策略参数 ====================
        # 仓位管理
        self.POSITION_SIZE = 50000  # 每笔交易金额
        self.MAX_POSITIONS = 20     # 最大持仓数量
        
        # 涨停相关参数
        self.LIMIT_UP_RATIO = 0.10  # 涨停幅度 10%
        
        # 卖出条件参数
        self.SELL_CONDITIONS = {
            'stop_loss_ratio': -0.10,      # 止损比例 -10%
            'take_profit_ratio': 0.20,     # 止盈比例 20%
            'max_hold_days': 30,           # 最大持有天数
            'trailing_stop_ratio': 0.05,   # 移动止损比例 5%
        }
        
        # ==================== 交易时间控制 ====================
        # 买入时间控制
        self.BUY_TIME_START = '09:30:00'
        self.BUY_TIME_END = '14:30:00'
        
        # 卖出时间控制  
        self.SELL_TIME_START = '09:30:00'
        self.SELL_TIME_END = '14:57:00'
        
        # ==================== 风险控制 ====================
        # 单日最大买入数量
        self.MAX_DAILY_BUYS = 5
        
        # 单只股票最大仓位比例
        self.MAX_SINGLE_POSITION_RATIO = 0.10  # 10%
        
        # 总仓位控制
        self.MAX_TOTAL_POSITION_RATIO = 0.95   # 95%
        
        # ==================== 数据配置 ====================
        # 股票池文件路径
        self.STOCK_POOL_FILE = '../all_targets_cache_5.csv'
        
        # 数据缓存目录
        self.DATA_CACHE_DIR = '../data_cache'
        
        # 是否使用缓存
        self.USE_CACHE = True
        
        # ==================== 输出配置 ====================
        # 结果输出目录
        self.OUTPUT_DIR = '../backtesting_results'
        
        # 是否保存交易记录
        self.SAVE_TRADES = True
        
        # 是否保存图表
        self.SAVE_PLOTS = True
        
        # 交易日志文件
        self.TRADE_LOG_FILE = 'backtesting_trades.xlsx'
        
        # ==================== 性能优化 ====================
        # 是否启用多进程
        self.ENABLE_MULTIPROCESSING = False
        
        # 进程数量
        self.NUM_PROCESSES = 4
        
        # 批处理大小
        self.BATCH_SIZE = 100
        
        # ==================== 日志配置 ====================
        # 日志级别
        self.LOG_LEVEL = 'INFO'
        
        # 日志文件
        self.LOG_FILE = 'backtesting.log'
        
        # 是否输出详细日志
        self.VERBOSE = True
        
        # ==================== 基准和风险 ====================
        # 基准指数
        self.BENCHMARK = '000300.SH'  # 沪深300
        
        # 无风险利率
        self.RISK_FREE_RATE = 0.03  # 3%
        
        # ==================== 策略特定参数 ====================
        # 首板策略相关参数
        self.FIRST_LIMIT_UP_PARAMS = {
            'min_volume_ratio': 2.0,       # 最小成交量倍数
            'min_turnover': 50000000,      # 最小成交额 5000万
            'exclude_st': True,            # 排除ST股票
            'exclude_new_stock_days': 60,  # 排除新股天数
            'ma5_filter': True,            # 是否使用MA5过滤
        }
        
        # 扫描缓存相关参数
        self.SCAN_CACHE_PARAMS = {
            'cache_file': 'all_targets_cache_5.csv',
            'update_interval': 1,          # 缓存更新间隔（天）
            'max_cache_age': 7,           # 最大缓存年龄（天）
        }
        
        # 确保输出目录存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.OUTPUT_DIR,
            self.DATA_CACHE_DIR,
            os.path.dirname(self.LOG_FILE) if os.path.dirname(self.LOG_FILE) else '.'
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
    
    def get_date_range(self):
        """
        获取日期范围
        
        Returns:
            tuple: (start_date, end_date) datetime对象
        """
        start_date = datetime.strptime(self.START_DATE, '%Y%m%d')
        end_date = datetime.strptime(self.END_DATE, '%Y%m%d')
        return start_date, end_date
    
    def update_dates(self, start_date: str, end_date: str):
        """
        更新回测日期范围
        
        Args:
            start_date: 开始日期 'YYYYMMDD'
            end_date: 结束日期 'YYYYMMDD'
        """
        self.START_DATE = start_date
        self.END_DATE = end_date
    
    def to_dict(self):
        """
        将配置转换为字典
        
        Returns:
            dict: 配置字典
        """
        config_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                config_dict[attr_name] = getattr(self, attr_name)
        return config_dict
    
    def from_dict(self, config_dict: dict):
        """
        从字典加载配置
        
        Args:
            config_dict: 配置字典
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def validate(self):
        """
        验证配置参数的有效性
        
        Returns:
            tuple: (is_valid, error_messages)
        """
        errors = []
        
        # 验证日期格式
        try:
            datetime.strptime(self.START_DATE, '%Y%m%d')
            datetime.strptime(self.END_DATE, '%Y%m%d')
        except ValueError:
            errors.append("日期格式错误，应为 'YYYYMMDD'")
        
        # 验证资金参数
        if self.INITIAL_CASH <= 0:
            errors.append("初始资金必须大于0")
        
        if self.POSITION_SIZE <= 0:
            errors.append("单笔交易金额必须大于0")
        
        # 验证比例参数
        if not (0 <= self.COMMISSION <= 1):
            errors.append("手续费比例应在0-1之间")
        
        if not (0 <= self.MAX_TOTAL_POSITION_RATIO <= 1):
            errors.append("总仓位比例应在0-1之间")
        
        # 验证持仓参数
        if self.MAX_POSITIONS <= 0:
            errors.append("最大持仓数量必须大于0")
        
        return len(errors) == 0, errors


class QuickConfigs:
    """快速配置预设"""
    
    @staticmethod
    def get_debug_config():
        """获取调试配置"""
        config = BacktestingConfig()
        config.START_DATE = '20241001'
        config.END_DATE = '20241031'
        config.INITIAL_CASH = 100000
        config.MAX_POSITIONS = 5
        config.VERBOSE = True
        return config
    
    @staticmethod
    def get_production_config():
        """获取生产配置"""
        config = BacktestingConfig()
        config.START_DATE = '20240101'
        config.END_DATE = '20241231'
        config.INITIAL_CASH = 1000000
        config.MAX_POSITIONS = 20
        config.VERBOSE = False
        return config
    
    @staticmethod
    def get_fast_test_config():
        """获取快速测试配置"""
        config = BacktestingConfig()
        config.START_DATE = '20241201'
        config.END_DATE = '20241231'
        config.INITIAL_CASH = 500000
        config.MAX_POSITIONS = 10
        config.USE_CACHE = True
        return config