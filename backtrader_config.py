"""
Backtrader 配置文件
集中管理回测参数和设置
"""

from datetime import datetime, timedelta


class BacktraderConfig:
    """Backtrader 回测配置类"""
    
    # 基本回测参数
    INITIAL_CASH = 1000000  # 初始资金
    COMMISSION = 0.0003     # 手续费率
    SLIPPAGE = 0.001       # 滑点
    
    # 时间设置
    START_DATE = '20240101'
    END_DATE = '20241231'
    
    # 策略参数
    POSITION_SIZE = 50000           # 每笔交易金额
    MAX_POSITIONS = 20              # 最大持仓数量
    SELL_MA5_DAYS = 3              # 跌破MA5卖出天数
    SELL_LOSS_PCT = 0.08           # 止损百分比
    LIMIT_UP_RATIO = 0.095         # 涨停比率
    LIMIT_DOWN_RATIO = -0.095      # 跌停比率
    
    # 时间控制
    BUY_START_TIME = '09:30:00'    # 买入开始时间
    BUY_END_TIME = '14:50:00'      # 买入结束时间
    SELL_START_TIME = '09:30:00'   # 卖出开始时间
    SELL_END_TIME = '14:50:00'     # 卖出结束时间
    
    # 基准和风险
    BENCHMARK_TICKER = '000300.SH'  # 基准指数
    RISK_FREE_RATE = 0.02          # 无风险利率
    
    # 数据设置
    DATA_FREQUENCY = 'daily'        # 数据频率: daily, minute
    STOCK_POOL_FILE = 'stock_pool.txt'  # 股票池文件
    
    # 选股缓存设置
    STOCK_SELECTION_CACHE_DIR = 'stock_selection_cache'
    USE_CACHE = True
    
    # 输出设置
    ENABLE_PLOTTING = True          # 启用绘图
    SAVE_RESULTS = True            # 保存结果
    RESULTS_DIR = 'backtest_results'
    
    # 性能优化
    PRELOAD = True                 # 预加载数据
    RUNONCE = True                 # 运行一次模式
    EXACTBARS = False              # 精确K线模式
    
    # 日志设置
    LOG_LEVEL = 'INFO'             # 日志级别
    LOG_TRADES = True              # 记录交易日志
    LOG_ORDERS = True              # 记录订单日志
    
    @classmethod
    def get_cerebro_params(cls):
        """获取 Cerebro 参数"""
        return {
            'preload': cls.PRELOAD,
            'runonce': cls.RUNONCE,
            'exactbars': cls.EXACTBARS
        }
    
    @classmethod
    def get_broker_params(cls):
        """获取 Broker 参数"""
        return {
            'cash': cls.INITIAL_CASH,
            'commission': cls.COMMISSION,
            'slip_perc': cls.SLIPPAGE
        }
    
    @classmethod
    def get_strategy_params(cls):
        """获取策略参数"""
        return {
            'position_size': cls.POSITION_SIZE,
            'max_positions': cls.MAX_POSITIONS,
            'sell_ma5_days': cls.SELL_MA5_DAYS,
            'sell_loss_pct': cls.SELL_LOSS_PCT,
            'limit_up_ratio': cls.LIMIT_UP_RATIO,
            'limit_down_ratio': cls.LIMIT_DOWN_RATIO,
            'buy_start_time': cls.BUY_START_TIME,
            'buy_end_time': cls.BUY_END_TIME,
            'sell_start_time': cls.SELL_START_TIME,
            'sell_end_time': cls.SELL_END_TIME,
            'benchmark_ticker': cls.BENCHMARK_TICKER,
            'risk_free_rate': cls.RISK_FREE_RATE
        }
    
    @classmethod
    def get_data_params(cls):
        """获取数据参数"""
        return {
            'start_date': cls.START_DATE,
            'end_date': cls.END_DATE,
            'frequency': cls.DATA_FREQUENCY,
            'stock_pool_file': cls.STOCK_POOL_FILE
        }


# 预定义配置模板
class QuickConfigs:
    """快速配置模板"""
    
    @staticmethod
    def small_capital_test():
        """小资金测试配置"""
        config = BacktraderConfig()
        config.INITIAL_CASH = 100000
        config.POSITION_SIZE = 10000
        config.MAX_POSITIONS = 10
        return config
    
    @staticmethod
    def large_capital_test():
        """大资金测试配置"""
        config = BacktraderConfig()
        config.INITIAL_CASH = 10000000
        config.POSITION_SIZE = 100000
        config.MAX_POSITIONS = 50
        return config
    
    @staticmethod
    def short_term_test():
        """短期测试配置"""
        config = BacktraderConfig()
        config.START_DATE = '20240101'
        config.END_DATE = '20240331'
        return config
    
    @staticmethod
    def conservative_strategy():
        """保守策略配置"""
        config = BacktraderConfig()
        config.SELL_LOSS_PCT = 0.05  # 更严格的止损
        config.MAX_POSITIONS = 10    # 更少的持仓
        config.POSITION_SIZE = 30000 # 更小的仓位
        return config
    
    @staticmethod
    def aggressive_strategy():
        """激进策略配置"""
        config = BacktraderConfig()
        config.SELL_LOSS_PCT = 0.12  # 更宽松的止损
        config.MAX_POSITIONS = 30    # 更多的持仓
        config.POSITION_SIZE = 80000 # 更大的仓位
        return config


# 环境配置
class EnvironmentConfig:
    """环境配置"""
    
    # 开发环境
    DEVELOPMENT = {
        'log_level': 'DEBUG',
        'enable_plotting': True,
        'save_results': True,
        'use_cache': True
    }
    
    # 生产环境
    PRODUCTION = {
        'log_level': 'INFO',
        'enable_plotting': False,
        'save_results': True,
        'use_cache': True
    }
    
    # 测试环境
    TESTING = {
        'log_level': 'WARNING',
        'enable_plotting': False,
        'save_results': False,
        'use_cache': False
    }


if __name__ == "__main__":
    # 配置示例
    print("Backtrader 配置示例")
    print("=" * 40)
    
    # 默认配置
    print("默认配置:")
    config = BacktraderConfig()
    print(f"初始资金: {config.INITIAL_CASH:,}")
    print(f"手续费率: {config.COMMISSION:.4f}")
    print(f"回测期间: {config.START_DATE} - {config.END_DATE}")
    
    # 快速配置示例
    print("\n小资金测试配置:")
    small_config = QuickConfigs.small_capital_test()
    print(f"初始资金: {small_config.INITIAL_CASH:,}")
    print(f"每笔交易: {small_config.POSITION_SIZE:,}")
    print(f"最大持仓: {small_config.MAX_POSITIONS}")