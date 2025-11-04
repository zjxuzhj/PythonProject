"""
Backtesting.py 框架性能分析器
提供详细的回测结果分析和可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class BacktestingAnalyzer:
    """回测结果分析器"""
    
    def __init__(self, results: Dict, config=None):
        """
        初始化分析器
        
        Args:
            results: 回测结果字典
            config: 配置对象
        """
        self.results = results
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 提取数据
        self.portfolio_stats = results.get('portfolio_statistics', {})
        self.individual_results = results.get('individual_results', {})
        self.all_trades = results.get('all_trades', [])
        
        # 转换为DataFrame便于分析
        self.trades_df = pd.DataFrame(self.all_trades) if self.all_trades else pd.DataFrame()
        self.individual_df = self._create_individual_df()
    
    def _create_individual_df(self) -> pd.DataFrame:
        """创建个股结果DataFrame"""
        if not self.individual_results:
            return pd.DataFrame()
        
        data = []
        for stock_code, result in self.individual_results.items():
            data.append({
                'stock_code': stock_code,
                'total_return': result.get('total_return', 0),
                'total_trades': result.get('total_trades', 0),
                'win_rate': result.get('win_rate', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'final_value': result.get('final_value', 0)
            })
        
        return pd.DataFrame(data)
    
    def calculate_detailed_statistics(self) -> Dict:
        """计算详细统计指标"""
        try:
            stats = {}
            
            # 基础统计
            if not self.individual_df.empty:
                stats['basic'] = {
                    'total_stocks': len(self.individual_df),
                    'profitable_stocks': len(self.individual_df[self.individual_df['total_return'] > 0]),
                    'loss_stocks': len(self.individual_df[self.individual_df['total_return'] < 0]),
                    'zero_return_stocks': len(self.individual_df[self.individual_df['total_return'] == 0])
                }
                
                # 收益率统计
                returns = self.individual_df['total_return']
                stats['returns'] = {
                    'mean': returns.mean(),
                    'median': returns.median(),
                    'std': returns.std(),
                    'min': returns.min(),
                    'max': returns.max(),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'percentile_25': returns.quantile(0.25),
                    'percentile_75': returns.quantile(0.75)
                }
                
                # 交易统计
                trades = self.individual_df['total_trades']
                stats['trades'] = {
                    'total_trades': trades.sum(),
                    'avg_trades_per_stock': trades.mean(),
                    'max_trades_per_stock': trades.max(),
                    'stocks_with_trades': len(self.individual_df[self.individual_df['total_trades'] > 0])
                }
                
                # 胜率统计
                win_rates = self.individual_df[self.individual_df['total_trades'] > 0]['win_rate']
                if not win_rates.empty:
                    stats['win_rates'] = {
                        'avg_win_rate': win_rates.mean(),
                        'median_win_rate': win_rates.median(),
                        'min_win_rate': win_rates.min(),
                        'max_win_rate': win_rates.max()
                    }
                
                # 回撤统计
                drawdowns = self.individual_df['max_drawdown']
                stats['drawdowns'] = {
                    'avg_max_drawdown': drawdowns.mean(),
                    'worst_drawdown': drawdowns.max(),
                    'best_drawdown': drawdowns.min(),
                    'median_drawdown': drawdowns.median()
                }
            
            # 交易层面统计
            if not self.trades_df.empty:
                sell_trades = self.trades_df[self.trades_df['action'] == 'SELL']
                if not sell_trades.empty:
                    returns = sell_trades['return_rate'] * 100
                    profits = sell_trades['profit']
                    
                    stats['trade_level'] = {
                        'total_sell_trades': len(sell_trades),
                        'profitable_trades': len(sell_trades[sell_trades['return_rate'] > 0]),
                        'loss_trades': len(sell_trades[sell_trades['return_rate'] < 0]),
                        'avg_return_per_trade': returns.mean(),
                        'median_return_per_trade': returns.median(),
                        'max_return_per_trade': returns.max(),
                        'min_return_per_trade': returns.min(),
                        'total_profit': profits.sum(),
                        'avg_profit_per_trade': profits.mean(),
                        'largest_profit': profits.max(),
                        'largest_loss': profits.min()
                    }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"计算详细统计失败: {e}")
            return {}
    
    def analyze_monthly_performance(self) -> pd.DataFrame:
        """分析月度表现"""
        try:
            if self.trades_df.empty:
                return pd.DataFrame()
            
            # 转换日期
            self.trades_df['date'] = pd.to_datetime(self.trades_df['date'])
            self.trades_df['year_month'] = self.trades_df['date'].dt.to_period('M')
            
            # 按月统计
            monthly_stats = []
            
            for period in self.trades_df['year_month'].unique():
                month_trades = self.trades_df[self.trades_df['year_month'] == period]
                sell_trades = month_trades[month_trades['action'] == 'SELL']
                
                if not sell_trades.empty:
                    monthly_stat = {
                        'year_month': str(period),
                        'total_trades': len(sell_trades),
                        'profitable_trades': len(sell_trades[sell_trades['return_rate'] > 0]),
                        'win_rate': len(sell_trades[sell_trades['return_rate'] > 0]) / len(sell_trades) * 100,
                        'total_return': sell_trades['return_rate'].sum() * 100,
                        'avg_return': sell_trades['return_rate'].mean() * 100,
                        'total_profit': sell_trades['profit'].sum(),
                        'best_trade': sell_trades['return_rate'].max() * 100,
                        'worst_trade': sell_trades['return_rate'].min() * 100
                    }
                    monthly_stats.append(monthly_stat)
            
            return pd.DataFrame(monthly_stats)
            
        except Exception as e:
            self.logger.error(f"分析月度表现失败: {e}")
            return pd.DataFrame()
    
    def analyze_holding_period(self) -> Dict:
        """分析持仓周期"""
        try:
            if self.trades_df.empty:
                return {}
            
            # 匹配买卖交易
            buy_trades = self.trades_df[self.trades_df['action'] == 'BUY'].copy()
            sell_trades = self.trades_df[self.trades_df['action'] == 'SELL'].copy()
            
            holding_periods = []
            
            for _, sell_trade in sell_trades.iterrows():
                # 找到对应的买入交易
                matching_buys = buy_trades[
                    (buy_trades['stock_code'] == sell_trade['stock_code']) &
                    (buy_trades['date'] <= sell_trade['date'])
                ]
                
                if not matching_buys.empty:
                    buy_trade = matching_buys.iloc[-1]  # 取最近的买入
                    
                    buy_date = pd.to_datetime(buy_trade['date'])
                    sell_date = pd.to_datetime(sell_trade['date'])
                    holding_days = (sell_date - buy_date).days
                    
                    holding_periods.append({
                        'stock_code': sell_trade['stock_code'],
                        'buy_date': buy_date,
                        'sell_date': sell_date,
                        'holding_days': holding_days,
                        'return_rate': sell_trade['return_rate'] * 100
                    })
            
            if not holding_periods:
                return {}
            
            holding_df = pd.DataFrame(holding_periods)
            
            # 统计持仓周期
            stats = {
                'avg_holding_days': holding_df['holding_days'].mean(),
                'median_holding_days': holding_df['holding_days'].median(),
                'min_holding_days': holding_df['holding_days'].min(),
                'max_holding_days': holding_df['holding_days'].max(),
                'std_holding_days': holding_df['holding_days'].std()
            }
            
            # 按持仓周期分组分析收益
            holding_df['period_group'] = pd.cut(
                holding_df['holding_days'],
                bins=[0, 3, 7, 15, 30, float('inf')],
                labels=['1-3天', '4-7天', '8-15天', '16-30天', '30天以上']
            )
            
            period_analysis = holding_df.groupby('period_group').agg({
                'return_rate': ['count', 'mean', 'std'],
                'holding_days': 'mean'
            }).round(2)
            
            stats['period_analysis'] = period_analysis
            stats['holding_data'] = holding_df
            
            return stats
            
        except Exception as e:
            self.logger.error(f"分析持仓周期失败: {e}")
            return {}
    
    def create_performance_plots(self, save_dir: str = None):
        """创建性能图表"""
        try:
            if save_dir is None:
                save_dir = self.config.OUTPUT_DIR if self.config else 'backtesting_results'
            
            os.makedirs(save_dir, exist_ok=True)
            
            # 1. 收益率分布图
            self._plot_return_distribution(save_dir)
            
            # 2. 个股表现散点图
            self._plot_stock_performance_scatter(save_dir)
            
            # 3. 月度表现图
            self._plot_monthly_performance(save_dir)
            
            # 4. 持仓周期分析图
            self._plot_holding_period_analysis(save_dir)
            
            # 5. 交易统计图
            self._plot_trade_statistics(save_dir)
            
            self.logger.info(f"性能图表已保存到: {save_dir}")
            
        except Exception as e:
            self.logger.error(f"创建性能图表失败: {e}")
    
    def _plot_return_distribution(self, save_dir: str):
        """绘制收益率分布图"""
        if self.individual_df.empty:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 直方图
        returns = self.individual_df['total_return']
        ax1.hist(returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(returns.mean(), color='red', linestyle='--', label=f'平均值: {returns.mean():.2f}%')
        ax1.axvline(returns.median(), color='green', linestyle='--', label=f'中位数: {returns.median():.2f}%')
        ax1.set_xlabel('收益率 (%)')
        ax1.set_ylabel('频数')
        ax1.set_title('收益率分布直方图')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        ax2.boxplot(returns, vert=True)
        ax2.set_ylabel('收益率 (%)')
        ax2.set_title('收益率箱线图')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'return_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stock_performance_scatter(self, save_dir: str):
        """绘制个股表现散点图"""
        if self.individual_df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 创建散点图
        scatter = ax.scatter(
            self.individual_df['total_trades'],
            self.individual_df['total_return'],
            c=self.individual_df['win_rate'],
            s=self.individual_df['max_drawdown'].abs() * 5,  # 回撤越大，点越大
            alpha=0.6,
            cmap='RdYlGn'
        )
        
        # 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label('胜率 (%)')
        
        ax.set_xlabel('交易次数')
        ax.set_ylabel('总收益率 (%)')
        ax.set_title('个股表现散点图\n(颜色表示胜率，大小表示最大回撤)')
        ax.grid(True, alpha=0.3)
        
        # 添加零线
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'stock_performance_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_monthly_performance(self, save_dir: str):
        """绘制月度表现图"""
        monthly_df = self.analyze_monthly_performance()
        if monthly_df.empty:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 月度收益率
        ax1.bar(monthly_df['year_month'], monthly_df['total_return'], alpha=0.7, color='skyblue')
        ax1.set_title('月度总收益率')
        ax1.set_ylabel('收益率 (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 月度胜率
        ax2.plot(monthly_df['year_month'], monthly_df['win_rate'], marker='o', color='green')
        ax2.set_title('月度胜率')
        ax2.set_ylabel('胜率 (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 月度交易次数
        ax3.bar(monthly_df['year_month'], monthly_df['total_trades'], alpha=0.7, color='orange')
        ax3.set_title('月度交易次数')
        ax3.set_ylabel('交易次数')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 月度平均收益
        ax4.plot(monthly_df['year_month'], monthly_df['avg_return'], marker='s', color='red')
        ax4.set_title('月度平均收益率')
        ax4.set_ylabel('平均收益率 (%)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'monthly_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_holding_period_analysis(self, save_dir: str):
        """绘制持仓周期分析图"""
        holding_stats = self.analyze_holding_period()
        if not holding_stats or 'holding_data' not in holding_stats:
            return
        
        holding_df = holding_stats['holding_data']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 持仓天数分布
        ax1.hist(holding_df['holding_days'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax1.axvline(holding_stats['avg_holding_days'], color='red', linestyle='--', 
                   label=f'平均: {holding_stats["avg_holding_days"]:.1f}天')
        ax1.set_xlabel('持仓天数')
        ax1.set_ylabel('频数')
        ax1.set_title('持仓天数分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 持仓天数与收益率关系
        ax2.scatter(holding_df['holding_days'], holding_df['return_rate'], alpha=0.6, color='purple')
        ax2.set_xlabel('持仓天数')
        ax2.set_ylabel('收益率 (%)')
        ax2.set_title('持仓天数与收益率关系')
        ax2.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(holding_df['holding_days'], holding_df['return_rate'], 1)
        p = np.poly1d(z)
        ax2.plot(holding_df['holding_days'], p(holding_df['holding_days']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'holding_period_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_trade_statistics(self, save_dir: str):
        """绘制交易统计图"""
        if self.trades_df.empty:
            return
        
        sell_trades = self.trades_df[self.trades_df['action'] == 'SELL']
        if sell_trades.empty:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 盈亏分布
        profits = sell_trades['profit']
        ax1.hist(profits, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='-', alpha=0.5)
        ax1.axvline(profits.mean(), color='blue', linestyle='--', label=f'平均: {profits.mean():.2f}')
        ax1.set_xlabel('盈亏金额')
        ax1.set_ylabel('频数')
        ax1.set_title('单笔交易盈亏分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 收益率分布
        returns = sell_trades['return_rate'] * 100
        ax2.hist(returns, bins=30, alpha=0.7, color='gold', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='-', alpha=0.5)
        ax2.axvline(returns.mean(), color='blue', linestyle='--', label=f'平均: {returns.mean():.2f}%')
        ax2.set_xlabel('收益率 (%)')
        ax2.set_ylabel('频数')
        ax2.set_title('单笔交易收益率分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 累计收益曲线
        sell_trades_sorted = sell_trades.sort_values('date')
        cumulative_profit = sell_trades_sorted['profit'].cumsum()
        ax3.plot(range(len(cumulative_profit)), cumulative_profit, color='navy')
        ax3.set_xlabel('交易序号')
        ax3.set_ylabel('累计盈亏')
        ax3.set_title('累计盈亏曲线')
        ax3.grid(True, alpha=0.3)
        
        # 胜率统计
        profitable_count = len(sell_trades[sell_trades['return_rate'] > 0])
        loss_count = len(sell_trades[sell_trades['return_rate'] < 0])
        zero_count = len(sell_trades[sell_trades['return_rate'] == 0])
        
        labels = ['盈利', '亏损', '平手']
        sizes = [profitable_count, loss_count, zero_count]
        colors = ['green', 'red', 'gray']
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('交易结果分布')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'trade_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_analysis_report(self, save_path: str = None) -> str:
        """生成分析报告"""
        try:
            if save_path is None:
                save_path = os.path.join(
                    self.config.OUTPUT_DIR if self.config else 'backtesting_results',
                    'analysis_report.txt'
                )
            
            # 计算详细统计
            detailed_stats = self.calculate_detailed_statistics()
            monthly_performance = self.analyze_monthly_performance()
            holding_analysis = self.analyze_holding_period()
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("Backtesting.py 框架详细分析报告\n")
                f.write("=" * 80 + "\n\n")
                
                # 基础统计
                if 'basic' in detailed_stats:
                    basic = detailed_stats['basic']
                    f.write("基础统计:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"总股票数: {basic['total_stocks']}\n")
                    f.write(f"盈利股票数: {basic['profitable_stocks']}\n")
                    f.write(f"亏损股票数: {basic['loss_stocks']}\n")
                    f.write(f"盈利股票比例: {basic['profitable_stocks']/basic['total_stocks']*100:.2f}%\n\n")
                
                # 收益率统计
                if 'returns' in detailed_stats:
                    returns = detailed_stats['returns']
                    f.write("收益率统计:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"平均收益率: {returns['mean']:.2f}%\n")
                    f.write(f"中位数收益率: {returns['median']:.2f}%\n")
                    f.write(f"收益率标准差: {returns['std']:.2f}%\n")
                    f.write(f"最大收益率: {returns['max']:.2f}%\n")
                    f.write(f"最小收益率: {returns['min']:.2f}%\n")
                    f.write(f"25%分位数: {returns['percentile_25']:.2f}%\n")
                    f.write(f"75%分位数: {returns['percentile_75']:.2f}%\n\n")
                
                # 交易统计
                if 'trades' in detailed_stats:
                    trades = detailed_stats['trades']
                    f.write("交易统计:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"总交易次数: {trades['total_trades']}\n")
                    f.write(f"平均每股交易次数: {trades['avg_trades_per_stock']:.2f}\n")
                    f.write(f"最大单股交易次数: {trades['max_trades_per_stock']}\n")
                    f.write(f"有交易的股票数: {trades['stocks_with_trades']}\n\n")
                
                # 持仓周期分析
                if holding_analysis:
                    f.write("持仓周期分析:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"平均持仓天数: {holding_analysis['avg_holding_days']:.2f}天\n")
                    f.write(f"中位数持仓天数: {holding_analysis['median_holding_days']:.2f}天\n")
                    f.write(f"最短持仓天数: {holding_analysis['min_holding_days']}天\n")
                    f.write(f"最长持仓天数: {holding_analysis['max_holding_days']}天\n\n")
                
                # 月度表现
                if not monthly_performance.empty:
                    f.write("月度表现:\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"{'月份':<10} {'交易次数':<8} {'胜率':<8} {'总收益率':<10} {'平均收益率':<10}\n")
                    f.write("-" * 60 + "\n")
                    for _, row in monthly_performance.iterrows():
                        f.write(f"{row['year_month']:<10} {row['total_trades']:<8} "
                               f"{row['win_rate']:<7.1f}% {row['total_return']:<9.2f}% "
                               f"{row['avg_return']:<9.2f}%\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            self.logger.info(f"分析报告已保存: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"生成分析报告失败: {e}")
            return ""