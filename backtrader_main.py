"""
Backtrader 主执行脚本
整合数据适配器、策略、分析器等所有组件
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# 导入自定义模块
from backtrader_data_feed import MultiStockDataManager
from backtrader_strategy import FirstLimitUpStrategy
from backtrader_analyzers import (
    TradeAnalyzer, 
    TimeBasedTradeAnalyzer, 
    PerformanceAnalyzer,
    MonthlyReturnAnalyzer,
    DetailedReportAnalyzer,
    print_analysis_results
)


class BacktraderBacktester:
    """
    Backtrader 回测器
    整合所有组件，提供完整的回测功能
    """
    
    def __init__(self, 
                 initial_cash=1000000,
                 commission=0.0003,
                 start_date='20240101',
                 end_date='20241231',
                 stock_pool_file='stock_pool.txt'):
        """
        初始化回测器
        
        Args:
            initial_cash: 初始资金
            commission: 手续费率
            start_date: 开始日期
            end_date: 结束日期
            stock_pool_file: 股票池文件
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.start_date = start_date
        self.end_date = end_date
        self.stock_pool_file = stock_pool_file
        
        # 初始化 Cerebro
        self.cerebro = bt.Cerebro()
        
        # 数据管理器
        self.data_manager = MultiStockDataManager()
        
    def setup_cerebro(self):
        """配置 Cerebro 引擎"""
        # 设置初始资金
        self.cerebro.broker.setcash(self.initial_cash)
        
        # 设置手续费
        self.cerebro.broker.setcommission(commission=self.commission)
        
        # 添加策略
        self.cerebro.addstrategy(FirstLimitUpStrategy)
        
        # 添加分析器
        self.cerebro.addanalyzer(TradeAnalyzer, _name='trade_analyzer')
        self.cerebro.addanalyzer(TimeBasedTradeAnalyzer, _name='time_trade_analyzer')
        self.cerebro.addanalyzer(PerformanceAnalyzer, _name='performance')
        self.cerebro.addanalyzer(MonthlyReturnAnalyzer, _name='monthly_returns')
        self.cerebro.addanalyzer(DetailedReportAnalyzer, _name='detailed_report')
        
        # 添加内置分析器
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='bt_trade_analyzer')
        
        # 添加观察器
        self.cerebro.addobserver(bt.observers.Broker)
        self.cerebro.addobserver(bt.observers.Trades)
        self.cerebro.addobserver(bt.observers.BuySell)
        
    def load_stock_pool(self):
        """加载股票池"""
        if os.path.exists(self.stock_pool_file):
            with open(self.stock_pool_file, 'r', encoding='utf-8') as f:
                stocks = [line.strip() for line in f.readlines() if line.strip()]
            return stocks
        else:
            # 默认股票池（示例）
            return [
                '000001.SZ', '000002.SZ', '000858.SZ', '002415.SZ',
                '600000.SH', '600036.SH', '600519.SH', '000300.SH'
            ]
    
    def add_data_feeds(self, stock_list):
        """添加数据源"""
        print(f"正在加载 {len(stock_list)} 只股票的数据...")
        
        for i, stock_code in enumerate(stock_list):
            try:
                # 获取数据
                data_feed = self.data_manager.get_data_feed(
                    stock_code=stock_code,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                
                if data_feed is not None:
                    # 添加到 Cerebro
                    self.cerebro.adddata(data_feed, name=stock_code)
                    print(f"已加载 {stock_code} 数据 ({i+1}/{len(stock_list)})")
                else:
                    print(f"警告: 无法加载 {stock_code} 数据")
                    
            except Exception as e:
                print(f"加载 {stock_code} 数据时出错: {e}")
                continue
    
    def run_backtest(self):
        """运行回测"""
        print("\n开始运行回测...")
        print(f"初始资金: {self.initial_cash:,.2f}")
        print(f"回测期间: {self.start_date} - {self.end_date}")
        
        # 运行回测
        results = self.cerebro.run()
        
        # 获取最终价值
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value / self.initial_cash - 1) * 100
        
        print(f"\n回测完成!")
        print(f"最终价值: {final_value:,.2f}")
        print(f"总收益率: {total_return:.2f}%")
        
        return results
    
    def print_detailed_results(self, results):
        """打印详细结果"""
        if not results:
            print("没有回测结果")
            return
        
        # 打印自定义分析器结果
        print_analysis_results(self.cerebro)
        
        # 打印内置分析器结果
        strat = results[0]
        
        print("\n内置分析器结果:")
        print("-" * 40)
        
        # 夏普比率
        if hasattr(strat.analyzers, 'sharpe'):
            sharpe = strat.analyzers.sharpe.get_analysis()
            if 'sharperatio' in sharpe and sharpe['sharperatio'] is not None:
                print(f"夏普比率: {sharpe['sharperatio']:.3f}")
            else:
                print("夏普比率: N/A (数据不足)")
        
        # 最大回撤
        if hasattr(strat.analyzers, 'drawdown'):
            drawdown = strat.analyzers.drawdown.get_analysis()
            if hasattr(drawdown, 'max') and hasattr(drawdown.max, 'drawdown'):
                print(f"最大回撤: {drawdown.max.drawdown:.2f}%")
                print(f"最大回撤期间: {drawdown.max.len} 天")
            else:
                print("最大回撤: N/A")
        
        # 收益分析
        if hasattr(strat.analyzers, 'returns'):
            returns = strat.analyzers.returns.get_analysis()
            if 'rtot' in returns:
                print(f"总收益率: {returns['rtot']:.2%}")
            if 'ravg' in returns:
                print(f"平均收益率: {returns['ravg']:.2%}")
    
    def plot_results(self):
        """绘制结果图表"""
        try:
            print("\n正在生成图表...")
            # 配置绘图参数
            self.cerebro.plot(
                style='candlestick',  # K线样式
                barup='red',         # 上涨颜色
                bardown='green',     # 下跌颜色
                volume=True,         # 显示成交量
                iplot=False          # 不使用交互式绘图
            )
            print("图表已生成")
        except Exception as e:
            print(f"绘图时出错: {e}")
    
    def save_results_to_file(self, results, filename='backtest_results.txt'):
        """保存结果到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Backtrader 回测结果报告\n")
                f.write("=" * 50 + "\n\n")
                
                # 基本信息
                f.write(f"初始资金: {self.initial_cash:,.2f}\n")
                f.write(f"最终价值: {self.cerebro.broker.getvalue():,.2f}\n")
                f.write(f"总收益率: {(self.cerebro.broker.getvalue()/self.initial_cash-1)*100:.2f}%\n")
                f.write(f"回测期间: {self.start_date} - {self.end_date}\n\n")
                
                # 详细分析结果
                if results:
                    strat = results[0]
                    
                    # 交易统计
                    if hasattr(strat.analyzers, 'trade_analyzer'):
                        trade_stats = strat.analyzers.trade_analyzer.get_analysis()
                        f.write("交易统计:\n")
                        f.write(f"总交易次数: {trade_stats['total_trades']}\n")
                        f.write(f"胜率: {trade_stats['win_rate']:.2%}\n")
                        f.write(f"盈亏比: {trade_stats['profit_loss_ratio']:.2f}\n\n")
                    
                    # 绩效指标
                    if hasattr(strat.analyzers, 'performance'):
                        perf = strat.analyzers.performance.get_analysis()
                        f.write("绩效指标:\n")
                        f.write(f"年化收益率: {perf['annual_return']:.2%}\n")
                        f.write(f"最大回撤: {perf['max_drawdown']:.2%}\n")
                        f.write(f"夏普比率: {perf['sharpe_ratio']:.2f}\n")
                        f.write(f"波动率: {perf['volatility']:.2%}\n\n")
            
            print(f"结果已保存到 {filename}")
            
        except Exception as e:
            print(f"保存结果时出错: {e}")


def main():
    """主函数"""
    print("Backtrader 回测系统启动")
    print("=" * 50)
    
    # 创建回测器
    backtester = BacktraderBacktester(
        initial_cash=1000000,
        commission=0.0003,
        start_date='20250815',
        end_date='20251102'
    )
    
    # 配置 Cerebro
    backtester.setup_cerebro()
    
    # 加载股票池
    stock_list = backtester.load_stock_pool()
    print(f"股票池包含 {len(stock_list)} 只股票")
    
    # 添加数据源
    backtester.add_data_feeds(stock_list)
    
    # 运行回测
    results = backtester.run_backtest()
    
    # 打印详细结果
    backtester.print_detailed_results(results)
    
    # 保存结果
    backtester.save_results_to_file(results)
    
    # 绘制图表
    backtester.plot_results()
    
    print("\n回测完成!")


if __name__ == "__main__":
    main()