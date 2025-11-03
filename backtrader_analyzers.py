"""
Backtrader 自定义分析器
实现策略绩效评估的各种指标计算
"""

import backtrader as bt
import pandas as pd
import numpy as np
from collections import defaultdict


class TradeAnalyzer(bt.Analyzer):
    """
    交易分析器
    分析交易胜率、盈亏比等指标
    """
    
    def __init__(self):
        self.trades = []
        self.trade_count = 0
        
    def notify_trade(self, trade):
        """交易完成时调用"""
        if trade.isclosed:
            self.trades.append({
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,  # 包含手续费的盈亏
                'size': trade.size,
                'price': trade.price,
                'value': trade.value,
                'commission': trade.commission,
                'dtopen': trade.dtopen,
                'dtclose': trade.dtclose,
                'baropen': trade.baropen,
                'barclose': trade.barclose,
                'long': trade.long
            })
            self.trade_count += 1
    
    def get_analysis(self):
        """返回分析结果"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_loss_ratio': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'total_pnl': 0.0,
                'total_pnl_comm': 0.0
            }
        
        # 计算基本统计
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnlcomm'] > 0]
        losing_trades = [t for t in self.trades if t['pnlcomm'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        avg_profit = np.mean([t['pnlcomm'] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = abs(np.mean([t['pnlcomm'] for t in losing_trades])) if losing_trades else 0.0
        
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_pnl_comm = sum(t['pnlcomm'] for t in self.trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'total_pnl': total_pnl,
            'total_pnl_comm': total_pnl_comm,
            'trades_detail': self.trades
        }


class TimeBasedTradeAnalyzer(bt.Analyzer):
    """
    基于时间段的交易分析器
    分析不同买入时段的交易表现
    """
    
    def __init__(self):
        self.time_trades = defaultdict(list)
        
    def notify_trade(self, trade):
        """交易完成时调用"""
        if trade.isclosed:
            # 获取开仓时间并转换为 datetime 对象
            try:
                if isinstance(trade.dtopen, float):
                    # 如果是浮点数，转换为 datetime
                    open_datetime = bt.num2date(trade.dtopen)
                else:
                    # 如果已经是 datetime 对象
                    open_datetime = trade.dtopen
                    
                open_time = open_datetime.time()
            except Exception as e:
                print(f"转换开仓时间失败: {e}")
                return
            
            # 分类时间段
            if open_time >= pd.Timestamp('09:30:00').time() and open_time < pd.Timestamp('10:00:00').time():
                time_category = "09:30-10:00"
            elif open_time >= pd.Timestamp('10:00:00').time() and open_time < pd.Timestamp('11:30:00').time():
                time_category = "10:00-11:30"
            elif open_time >= pd.Timestamp('13:00:00').time() and open_time < pd.Timestamp('14:50:00').time():
                time_category = "13:00-14:50"
            else:
                time_category = "Other"
            
            self.time_trades[time_category].append({
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'dtopen': trade.dtopen,
                'dtclose': trade.dtclose
            })
    
    def get_analysis(self):
        """返回按时间段分析的结果"""
        result = {}
        
        for time_category, trades in self.time_trades.items():
            if not trades:
                result[time_category] = {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'profit_loss_ratio': 0.0,
                    'avg_profit': 0.0,
                    'avg_loss': 0.0,
                    'total_pnl': 0.0
                }
                continue
            
            total_trades = len(trades)
            winning_trades = [t for t in trades if t['pnlcomm'] > 0]
            losing_trades = [t for t in trades if t['pnlcomm'] < 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
            
            avg_profit = np.mean([t['pnlcomm'] for t in winning_trades]) if winning_trades else 0.0
            avg_loss = abs(np.mean([t['pnlcomm'] for t in losing_trades])) if losing_trades else 0.0
            
            profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')
            total_pnl = sum(t['pnlcomm'] for t in trades)
            
            result[time_category] = {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl
            }
        
        return result


class PerformanceAnalyzer(bt.Analyzer):
    """
    绩效分析器
    计算夏普比率、最大回撤、Alpha、Beta等指标
    """
    
    params = (
        ('risk_free_rate', 0.02),  # 无风险利率
        ('benchmark_ticker', '000300.SH'),  # 基准指数
    )
    
    def __init__(self):
        self.portfolio_values = []
        self.dates = []
        
    def next(self):
        """每个交易日调用"""
        self.portfolio_values.append(self.strategy.broker.getvalue())
        self.dates.append(self.strategy.datas[0].datetime.date(0))
    
    def get_analysis(self):
        """返回绩效分析结果"""
        if len(self.portfolio_values) < 2:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'volatility': 0.0,
                'calmar_ratio': 0.0
            }
        
        # 转换为 pandas Series
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        
        # 计算日收益率
        daily_returns = portfolio_series.pct_change().dropna()
        
        # 总收益率
        total_return = (self.portfolio_values[-1] / self.portfolio_values[0]) - 1
        
        # 年化收益率
        trading_days = len(daily_returns)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        
        # 波动率（年化）
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # 最大回撤
        cumulative = (1 + daily_returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        # 夏普比率
        if volatility > 0:
            daily_risk_free = self.params.risk_free_rate / 252
            excess_returns = daily_returns - daily_risk_free
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # 卡尔马比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'calmar_ratio': calmar_ratio,
            'portfolio_values': portfolio_series,
            'daily_returns': daily_returns
        }


class MonthlyReturnAnalyzer(bt.Analyzer):
    """
    月度收益分析器
    计算每月收益率
    """
    
    def __init__(self):
        self.portfolio_values = []
        self.dates = []
        
    def next(self):
        """每个交易日调用"""
        self.portfolio_values.append(self.strategy.broker.getvalue())
        self.dates.append(self.strategy.datas[0].datetime.date(0))
    
    def get_analysis(self):
        """返回月度收益分析结果"""
        if len(self.portfolio_values) < 2:
            return {'monthly_returns': {}}
        
        # 转换为 pandas Series，确保索引是 DatetimeIndex
        try:
            # 将 date 对象转换为 datetime 对象
            datetime_index = pd.to_datetime(self.dates)
            portfolio_series = pd.Series(self.portfolio_values, index=datetime_index)
            
            # 计算月末值
            monthly_end_values = portfolio_series.resample('ME').last()
            
            # 计算月度收益率
            monthly_returns = monthly_end_values.pct_change().dropna()
            
            # 格式化输出
            monthly_return_dict = {}
            for date, return_rate in monthly_returns.items():
                month_str = f"{date.year}年{date.month}月"
                monthly_return_dict[month_str] = return_rate
            
            return {
                'monthly_returns': monthly_return_dict,
                'monthly_values': monthly_end_values
            }
            
        except Exception as e:
            print(f"月度收益分析失败: {e}")
            return {'monthly_returns': {}}


class DetailedReportAnalyzer(bt.Analyzer):
    """
    详细报告分析器
    生成综合性的策略报告
    """
    
    def __init__(self):
        self.start_value = None
        self.end_value = None
        self.start_date = None
        self.end_date = None
        
    def start(self):
        """回测开始时调用"""
        self.start_value = self.strategy.broker.getvalue()
        self.start_date = self.strategy.datas[0].datetime.date(0)
    
    def stop(self):
        """回测结束时调用"""
        self.end_value = self.strategy.broker.getvalue()
        self.end_date = self.strategy.datas[0].datetime.date(-1)
    
    def get_analysis(self):
        """返回详细报告"""
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'start_value': self.start_value,
            'end_value': self.end_value,
            'total_pnl': self.end_value - self.start_value if self.start_value else 0,
            'total_return_pct': ((self.end_value / self.start_value) - 1) * 100 if self.start_value else 0
        }


def print_analysis_results(cerebro):
    """
    打印所有分析器的结果
    
    Args:
        cerebro: Backtrader Cerebro 实例
    """
    print("\n" + "="*60)
    print("策略绩效评估报告")
    print("="*60)
    
    # 获取所有分析器结果
    analyzers = cerebro.runstrats[0][0].analyzers
    
    # 详细报告
    if hasattr(analyzers, 'detailed_report'):
        detail = analyzers.detailed_report.get_analysis()
        print(f"\n基本信息:")
        print(f"回测期间: {detail['start_date']} 至 {detail['end_date']}")
        print(f"初始资金: {detail['start_value']:,.2f}")
        print(f"最终价值: {detail['end_value']:,.2f}")
        print(f"总盈亏: {detail['total_pnl']:,.2f} ({detail['total_return_pct']:.2f}%)")
    
    # 交易统计
    if hasattr(analyzers, 'trade_analyzer'):
        trade_stats = analyzers.trade_analyzer.get_analysis()
        print(f"\n交易统计:")
        print(f"总交易次数: {trade_stats['total_trades']}")
        print(f"胜率: {trade_stats['win_rate']:.2%}")
        print(f"盈亏比: {trade_stats['profit_loss_ratio']:.2f}")
        print(f"平均盈利: {trade_stats['avg_profit']:.2f}")
        print(f"平均亏损: {trade_stats['avg_loss']:.2f}")
    
    # 时间段分析
    if hasattr(analyzers, 'time_trade_analyzer'):
        time_stats = analyzers.time_trade_analyzer.get_analysis()
        print(f"\n按买入时段分析:")
        for time_period, stats in time_stats.items():
            if stats['total_trades'] > 0:
                print(f"时段 [{time_period}]: "
                      f"交易次数: {stats['total_trades']}, "
                      f"胜率: {stats['win_rate']:.2%}, "
                      f"盈亏比: {stats['profit_loss_ratio']:.2f}")
    
    # 绩效指标
    if hasattr(analyzers, 'performance'):
        perf = analyzers.performance.get_analysis()
        print(f"\n绩效指标:")
        print(f"年化收益率: {perf['annual_return']:.2%}")
        print(f"最大回撤: {perf['max_drawdown']:.2%}")
        print(f"夏普比率: {perf['sharpe_ratio']:.2f}")
        print(f"波动率: {perf['volatility']:.2%}")
        print(f"卡尔马比率: {perf['calmar_ratio']:.2f}")
    
    # 月度收益
    if hasattr(analyzers, 'monthly_returns'):
        monthly = analyzers.monthly_returns.get_analysis()
        if monthly['monthly_returns']:
            print(f"\n月度收益率:")
            monthly_strs = []
            for month, return_rate in monthly['monthly_returns'].items():
                monthly_strs.append(f"{month} {return_rate:.2%}")
            print(", ".join(monthly_strs))
    
    print("\n" + "="*60)