"""
Backtesting.py 框架主执行文件
完整迁移自 qmt_real_time_backtest.py 的回测逻辑
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
import logging
import os
from datetime import datetime, timedelta, time
import json

# 导入原始模块
try:
    import first_limit_up_ma5_normal_scan as scan
    import getAllStockCsv
    from common_sell_logic import get_sell_decision, MarketDataContext
    from stock_info import StockInfo
    import xtquant.xtdata as xtdata
except ImportError as e:
    logging.warning(f"导入模块失败: {e}")

from backtesting_config import BacktestingConfig
from backtesting_data_manager import DataManager


class BacktestingRunner:
    """
    完整迁移自原始 Backtester 类的回测运行器
    实现与原始回测完全一致的逻辑
    """
    
    def __init__(self, start_date: str, end_date: str, 
                 initial_capital: float = 200000.0, 
                 position_size: float = 20000.0,
                 commission: float = 0.0,
                 min_commission: float = 0.0,
                 slippage: float = 0.0,
                 slippage_mode: str = "both"):
        """
        初始化回测运行器
        
        Args:
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'
            initial_capital: 初始资金
            position_size: 单笔交易金额
        """
        # 基本参数
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.position_size = position_size
        # 交易成本参数（与 Backtesting 框架风格一致）
        # commission: 比例(0.001=0.1%)或固定值(>=1)，每笔交易收取
        # min_commission: 最低手续费（元）
        # slippage: 滑点比例(0.005=0.5%)，买入上浮、卖出下浮
        # slippage_mode: 滑点应用方式："open"(仅开盘/买入价)、"close"(仅收盘/卖出价)、"both"(两者)
        self.commission = float(commission)
        self.min_commission = float(min_commission)
        self.slippage = float(slippage)
        self.slippage_mode = str(slippage_mode).lower()
        
        # 策略参数
        self.LIMIT_RATE = 0.10
        self.BENCHMARK_TICKER = "000300.SH"
        self.RISK_FREE_RATE = 0.03
        self.BUY_CUTOFF_TIME = time(14, 45)
        
        # 交易状态
        self.cash = initial_capital
        self.positions = {}  # {stock_code: {shares, buy_price, buy_date, hold_days, buy_time_category}}
        self.trade_log = []
        self.daily_values = []
        
        # 初始化工具
        try:
            self.query_tool = getAllStockCsv.StockQuery()
        except Exception as e:
            logging.warning(f"初始化股票查询工具失败: {e}")
            self.query_tool = None
        
        # 初始化数据管理器
        config = BacktestingConfig()
        config.START_DATE = start_date
        config.END_DATE = end_date
        self.data_manager = DataManager(config)
        
        # 设置日志
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"回测初始化完成: {start_date} - {end_date}, 初始资金: {initial_capital}; "
            f"commission={self.commission}, min_commission={self.min_commission}, slippage={self.slippage}, mode={self.slippage_mode}"
        )
    
    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('backtest.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        return self.data_manager.get_daily_data(stock_code, start_date, end_date)
    
    def get_daily_data_with_metrics(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取包含技术指标的日线数据"""
        return self.data_manager.get_daily_data_with_metrics(stock_code, start_date, end_date)
    
    def get_minute_data(self, stock_code: str, target_date: str) -> pd.DataFrame:
        """获取分钟数据"""
        return self.data_manager.get_minute_data(stock_code, target_date)
    
    def get_target_stocks_from_cache(self, date_str: str) -> List[str]:
        """从缓存获取目标股票"""
        return self.data_manager.get_target_stocks_from_cache(date_str)
    
    def calculate_buy_target_price(self, stock_code: str, current_date_str: str) -> Optional[float]:
        """计算买入目标价格"""
        return self.data_manager.calculate_buy_target_price(stock_code, current_date_str)
    
    def _get_buy_time_category(self, buy_datetime: datetime) -> str:
        """获取买入时间分类（与 qmt_real_time_backtest 对齐）"""
        t = buy_datetime.time()
        if time(9, 30) <= t < time(10, 0):
            return "09:30-10:00"
        elif time(10, 0) <= t < time(11, 30):
            return "10:00-11:30"
        elif time(13, 0) <= t < time(14, 50):
            return "13:00-14:50"
        else:
            return "Other"
    
    def log_trade(self, date_time: datetime, stock_code: str, action: str, 
                  shares: int, price: float, reason: str = "", pnl: Optional[float] = None, 
                  buy_time_category: Optional[str] = None, price_type: Optional[str] = None, ref_price: Optional[float] = None,
                  fee: Optional[float] = None):
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
            'buy_time_category': buy_time_category,
            'price_type': price_type or "",
            'ref_price': ref_price if ref_price is not None else "",
            'fee': fee if fee is not None else 0.0
        }
        self.trade_log.append(trade_record)
        
        action_desc = "买入" if action == "BUY" else "卖出"
        pnl_desc = f", 盈亏: {pnl:.2f}" if pnl is not None else ""
        print(f"  - {action_desc} {trade_record['stock_name']} {shares}股 @ {price:.2f}{pnl_desc}")

    def _calc_commission(self, gross_amount: float) -> float:
        """计算每笔交易手续费。支持比例或固定值，并应用最低手续费。"""
        if gross_amount <= 0:
            return 0.0
        fee = 0.0
        if self.commission <= 0:
            fee = 0.0
        elif self.commission >= 1.0:
            fee = self.commission
        else:
            fee = gross_amount * self.commission
        if self.min_commission and self.min_commission > 0:
            fee = max(fee, self.min_commission)
        return round(fee, 2)

    def _apply_slippage(self, price: float, action: str, price_type: Optional[str] = None) -> float:
        """根据滑点设置调整成交价。买入加价，卖出降价。"""
        if self.slippage <= 0 or price <= 0:
            return price
        mode = self.slippage_mode
        apply = False
        if mode == "both":
            apply = True
        elif mode == "open" and (action == "BUY"):
            # 买入通常对应开盘或盘中价格，这里按 BUY 应用
            apply = True if (price_type in (None, "开盘价", "预估买入价")) else True
        elif mode == "close" and (action == "SELL"):
            apply = True
        if not apply:
            return price
        if action == "BUY":
            return round(price * (1 + self.slippage), 2)
        else:
            return round(price * (1 - self.slippage), 2)
    
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
    
    def calculate_and_print_statistics(self):
        """计算并打印统计信息"""
        if not self.trade_log:
            print("没有交易记录")
            return {}
        
        # 转换为DataFrame便于分析
        trades_df = pd.DataFrame(self.trade_log)
        
        # 分离买入和卖出记录
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        print("\n=== 交易统计 ===")
        print(f"总交易次数: {len(trades_df)}")
        print(f"买入次数: {len(buy_trades)}")
        print(f"卖出次数: {len(sell_trades)}")
        
        if len(sell_trades) > 0:
            # 计算盈亏统计
            total_pnl = sell_trades['pnl'].sum()
            win_trades = sell_trades[sell_trades['pnl'] > 0]
            lose_trades = sell_trades[sell_trades['pnl'] <= 0]
            
            win_rate = len(win_trades) / len(sell_trades) * 100
            avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
            avg_loss = lose_trades['pnl'].mean() if len(lose_trades) > 0 else 0
            
            print(f"总盈亏: {total_pnl:.2f}")
            print(f"胜率: {win_rate:.2f}%")
            print(f"平均盈利: {avg_win:.2f}")
            print(f"平均亏损: {avg_loss:.2f}")
            
            # 按买入时段统计
            if 'buy_time_category' in sell_trades.columns:
                print("\n=== 按买入时段统计 ===")
                time_stats = sell_trades.groupby('buy_time_category').agg({
                    'pnl': ['count', 'sum', 'mean'],
                }).round(2)
                print(time_stats)
        
        # 计算收益率和风险指标
        if self.daily_values:
            daily_df = pd.DataFrame(self.daily_values)
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            # 按日期排序，确保月度统计的边界准确
            daily_df = daily_df.set_index('date').sort_index()
            
            # 计算日收益率
            daily_df['returns'] = daily_df['total_value'].pct_change()
            
            # 计算最大回撤
            daily_df['cummax'] = daily_df['total_value'].cummax()
            daily_df['drawdown'] = (daily_df['total_value'] - daily_df['cummax']) / daily_df['cummax']
            max_drawdown = daily_df['drawdown'].min()
            
            # 计算年化收益率
            total_return = (daily_df['total_value'].iloc[-1] - self.initial_capital) / self.initial_capital
            days = len(daily_df)
            annual_return = (1 + total_return) ** (252 / days) - 1
            
            # 计算夏普比率
            annual_volatility = daily_df['returns'].std() * np.sqrt(252)
            sharpe_ratio = (annual_return - self.RISK_FREE_RATE) / annual_volatility if annual_volatility > 0 else 0
            
            print(f"\n=== 收益与风险指标 ===")
            print(f"总收益率: {total_return:.2%}")
            print(f"年化收益率: {annual_return:.2%}")
            print(f"最大回撤: {max_drawdown:.2%}")
            print(f"年化波动率: {annual_volatility:.2%}")
            print(f"夏普比率: {sharpe_ratio:.2f}")
            
            # 与基准比较（如果有基准数据）
            try:
                benchmark_data = self.get_daily_data(self.BENCHMARK_TICKER, self.start_date, self.end_date)
                if not benchmark_data.empty:
                    benchmark_return = (benchmark_data['close'].iloc[-1] - benchmark_data['close'].iloc[0]) / benchmark_data['close'].iloc[0]
                    alpha = total_return - benchmark_return
                    print(f"基准收益率: {benchmark_return:.2%}")
                    print(f"Alpha: {alpha:.2%}")
            except Exception as e:
                self.logger.warning(f"计算基准比较失败: {e}")

            # 按自然月统计涨幅（1-31为自然月边界）
            try:
                monthly_groups = daily_df['total_value'].groupby(daily_df.index.to_period('M')).agg(['first', 'last'])
                monthly_groups['monthly_pct'] = (monthly_groups['last'] / monthly_groups['first'] - 1) * 100

                print("\n=== 按自然月统计涨幅 ===")
                header = f"{'月份':<10} {'当月起始净值':>12} {'当月结束净值':>12} {'月涨幅(%)':>10}"
                print(header)
                print("-" * len(header))
                for period, row in monthly_groups.iterrows():
                    month_str = period.strftime('%Y-%m')
                    start_val = row['first']
                    end_val = row['last']
                    pct = row['monthly_pct']
                    print(f"{month_str:<10} {start_val:>12.2f} {end_val:>12.2f} {pct:>10.2f}")
            except Exception as e:
                self.logger.warning(f"月度涨幅统计失败: {e}")
        
        return {
            'total_trades': len(trades_df),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_pnl': sell_trades['pnl'].sum() if len(sell_trades) > 0 else 0,
            'win_rate': len(sell_trades[sell_trades['pnl'] > 0]) / len(sell_trades) * 100 if len(sell_trades) > 0 else 0,
            'max_drawdown': max_drawdown if 'max_drawdown' in locals() else 0,
            'annual_return': annual_return if 'annual_return' in locals() else 0,
            'sharpe_ratio': sharpe_ratio if 'sharpe_ratio' in locals() else 0
        }
    
    def run(self):
        """运行回测 - 完整迁移自原始 Backtester.run()"""
        print(f"开始回测: {self.start_date} - {self.end_date}")
        print(f"初始资金: {self.initial_capital}")
        print(f"单笔交易金额: {self.position_size}")
        
        current_dt = datetime.strptime(self.start_date, "%Y%m%d")
        end_dt = datetime.strptime(self.end_date, "%Y%m%d")
        
        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y%m%d")
            print(f"\n--- 交易日: {current_dt.strftime('%Y-%m-%d')} ---")
            print(f"交易前现金: {self.cash:.2f}")
            
            # 检查是否为周末
            if current_dt.weekday() >= 5:
                print("周末，跳过。")
                current_dt += timedelta(days=1)
                continue
            
            # 检查是否为交易日
            is_trading_day_df = self.get_daily_data(self.BENCHMARK_TICKER, date_str, date_str)
            if is_trading_day_df.empty:
                print("节假日或非交易日，跳过。")
                current_dt += timedelta(days=1)
                continue
            
            # 1. 处理买入操作
            print("正在扫描新的可买入股票...")
            target_stocks = self.get_target_stocks_from_cache(date_str)
            if not target_stocks:
                print("今天没有发现目标股票。")
            else:
                potential_buys = []
                late_opportunities: List[Dict] = []
                for stock_code in target_stocks:
                    if self.cash < self.position_size:
                        break
                    
                    buy_target_price = self.calculate_buy_target_price(stock_code, date_str)
                    if buy_target_price is None:
                        continue
                    
                    # 获取分钟与当日开盘数据，用于优先采用开盘价买入
                    min_df = self.get_minute_data(stock_code, date_str)
                    day_df = self.get_daily_data(stock_code, date_str, date_str)

                    # 优先策略：当日开盘价<=预估买入价，则采用开盘价作为实际买入价格
                    open_candidate_added = False
                    day_open: Optional[float] = None
                    open_dt: Optional[datetime] = None
                    if not day_df.empty and 'open' in day_df.columns:
                        try:
                            day_open = float(day_df.iloc[-1]['open'])
                        except Exception:
                            day_open = None
                    # 计算开盘时间（优先使用分钟数据的首条时间；否则退化为09:30）
                    try:
                        if not min_df.empty:
                            open_dt = min_df.index.min().to_pydatetime()
                        else:
                            open_dt = datetime.combine(current_dt.date(), time(9, 30))
                    except Exception:
                        open_dt = datetime.combine(current_dt.date(), time(9, 30))

                    if (day_open is not None) and (day_open <= buy_target_price) and (open_dt and open_dt.time() < self.BUY_CUTOFF_TIME):
                        potential_buys.append({
                            'time': open_dt,
                            'stock': stock_code,
                            'price': day_open,
                            'price_type': '开盘价',
                            'ref_price': buy_target_price
                        })
                        open_candidate_added = True

                    # 原有逻辑：分钟触价采用预估买入价（作为参考基准）
                    if not min_df.empty:
                        touched_df = min_df[(min_df['low'] <= buy_target_price) & (min_df['high'] >= buy_target_price)]
                        if not touched_df.empty:
                            first_touch_time = touched_df.index[0]
                            if first_touch_time.time() < self.BUY_CUTOFF_TIME and not open_candidate_added:
                                potential_buys.append({
                                    'time': first_touch_time,
                                    'stock': stock_code,
                                    'price': buy_target_price,
                                    'price_type': '预估买入价',
                                    'ref_price': buy_target_price
                                })
                            elif first_touch_time.time() >= self.BUY_CUTOFF_TIME:
                                # 收集到当天的人类可读清单（晚于截止时间）
                                planned_shares = math.floor((self.position_size / buy_target_price) / 100) * 100
                                if planned_shares > 0:
                                    late_opportunities.append({
                                        'time': first_touch_time,
                                        'stock': stock_code,
                                        'name': self.query_tool.get_name_by_code(stock_code) if self.query_tool else stock_code,
                                        'price': buy_target_price,
                                        'shares': int(planned_shares),
                                        'amount': float(planned_shares * buy_target_price),
                                        'price_type': '预估买入价',
                                        'ref_price': buy_target_price
                                    })
                
                # 按时间排序
                potential_buys.sort(key=lambda x: x['time'])
                print(f"在 {self.BUY_CUTOFF_TIME.strftime('%H:%M')} 前发现 {len(potential_buys)} 个潜在买入机会。")
                
                skipped_buys = []
                
                # 执行买入
                for buy_order in potential_buys:
                    if buy_order['stock'] in self.positions:
                        print(f"  - [跳过] 股票 {buy_order['stock']} 已在持仓中，本次不再买入。")
                        continue
                    
                    if self.cash >= self.position_size:
                        raw_price = buy_order['price']
                        price_type = buy_order.get('price_type', '预估买入价')
                        ref_price = buy_order.get('ref_price', raw_price)
                        # 应用滑点后的实际成交价
                        buy_price = self._apply_slippage(raw_price, action="BUY", price_type=price_type)
                        shares_to_buy = math.floor((self.position_size / buy_price) / 100) * 100
                        
                        if shares_to_buy > 0:
                            gross = shares_to_buy * buy_price
                            fee = self._calc_commission(gross)
                            cost = gross + fee
                            buy_time_cat = self._get_buy_time_category(buy_order['time'])
                            
                            self.cash -= cost
                            self.positions[buy_order['stock']] = {
                                'shares': shares_to_buy,
                                'buy_price': buy_price,
                                'buy_date': date_str,
                                'hold_days': 0,
                                'buy_time_category': buy_time_cat,
                                'buy_commission': fee
                            }
                            
                            self.log_trade(
                                buy_order['time'], buy_order['stock'], "BUY",
                                shares_to_buy, buy_price, buy_time_category=buy_time_cat,
                                price_type=price_type, ref_price=ref_price, fee=fee
                            )
                        else:
                            print(f"  - [跳过] 资金不足以购买至少100股的 {buy_order['stock']} (价格: {buy_price:.2f})。")
                            skipped_buys.append(self.query_tool.get_name_by_code(buy_order['stock']) if self.query_tool else buy_order['stock'])
                    else:
                        skipped_buys.append(self.query_tool.get_name_by_code(buy_order['stock']) if self.query_tool else buy_order['stock'])

                # 输出晚于截止时间的潜在买入机会（与 qmt_real_time_backtest 保持一致的可读格式）
                if late_opportunities:
                    print("今日可买入但买入时间在14:45后的股票清单：")
                    for op in late_opportunities:
                        print(
                            f"  [14:45后] {op['time'].strftime('%H:%M')} 可买: {op['name']} | {op['shares']} 股 @ {op['price']:.2f} | 金额: {op['amount']:.2f} | 价格类型: {op['price_type']} | 参考买入价: {op['ref_price']:.2f} | 原因: 触价时间晚于14:45"
                        )

                if skipped_buys:
                    print("今日现金不足，无法进行更多买入。")
                    print(f"  - 当日因现金不足错过的买入机会共 {len(skipped_buys)} 个。")
                    print(f"  - 错过股票: {', '.join(skipped_buys)}")
            
            # 2. 处理卖出操作
            positions_to_check = list(self.positions.keys())
            if not positions_to_check:
                print("没有需要卖出的持仓。")
            else:
                print("正在检查持仓的卖出信号...")
                for stock_code in positions_to_check:
                    position = self.positions[stock_code]
                    
                    # 当天买入的不卖出
                    if position['buy_date'] == date_str:
                        continue
                    
                    # 更新持仓天数
                    position['hold_days'] += 1
                    
                    # 检查卖出条件
                    sell_price, sell_reason = self.check_sell_conditions(stock_code, position, current_dt)
                    
                    if sell_price is not None:
                        shares_to_sell = position['shares']
                        buy_price = position['buy_price']
                        # 卖出按收盘价应用滑点
                        exec_sell_price = self._apply_slippage(sell_price, action="SELL", price_type="收盘价")
                        gross_proceeds = shares_to_sell * exec_sell_price
                        sell_fee = self._calc_commission(gross_proceeds)
                        net_proceeds = gross_proceeds - sell_fee
                        self.cash += net_proceeds
                        # PnL 包含双边手续费
                        buy_fee = position.get('buy_commission', 0.0)
                        profit_loss = (exec_sell_price - buy_price) * shares_to_sell - buy_fee - sell_fee

                        self.log_trade(
                            date_time=current_dt,
                            stock_code=stock_code,
                            action="SELL",
                            shares=shares_to_sell,
                            price=exec_sell_price,
                            reason=sell_reason,
                            pnl=profit_loss,
                            buy_time_category=position['buy_time_category'],
                            price_type="收盘价",
                            ref_price=sell_price,
                            fee=sell_fee
                        )
                        
                        del self.positions[stock_code]
            
            # 3. 记录每日价值
            holdings_value = 0
            daily_holdings_details = []
            
            for stock_code, position in self.positions.items():
                eod_df = self.get_daily_data_with_metrics(stock_code, date_str, date_str)
                last_close = eod_df.iloc[-1]['close'] if not eod_df.empty else position['buy_price']
                
                shares = position['shares']
                buy_price = position['buy_price']
                market_value = shares * last_close
                pnl = (last_close - buy_price) * shares
                
                daily_holdings_details.append({
                    "code": stock_code,
                    "shares": shares,
                    "close": last_close,
                    "value": market_value,
                    "pnl": pnl
                })
                holdings_value += market_value
            
            total_value = self.cash + holdings_value
            
            # 记录每日价值
            self.daily_values.append({
                'date': date_str,
                'cash': self.cash,
                'holdings_value': holdings_value,
                'total_value': total_value,
                'positions_count': len(self.positions),
                'holdings_details': daily_holdings_details
            })
            
            print(f"当日收盘总资产: {total_value:.2f} (现金: {self.cash:.2f}, 持仓市值: {holdings_value:.2f})")
            print(f"持仓数量: {len(self.positions)}")
            
            current_dt += timedelta(days=1)
        
        # 回测结束，计算统计信息
        print("\n" + "="*50)
        print("回测完成")
        print("="*50)
        
        stats = self.calculate_and_print_statistics()
        
        # 保存结果
        self.save_results()
        
        return stats
    
    def save_results(self):
        """保存回测结果"""
        try:
            # 保存交易记录
            if self.trade_log:
                trades_df = pd.DataFrame(self.trade_log)
                trades_df.to_csv('trade_log.csv', index=False, encoding='utf-8')
                print(f"交易记录已保存到 trade_log.csv")
            
            # 保存每日价值
            if self.daily_values:
                daily_df = pd.DataFrame(self.daily_values)
                daily_df.to_csv('daily_values.csv', index=False, encoding='utf-8')
                print(f"每日价值已保存到 daily_values.csv")
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")


def main():
    """主函数"""
    # 回测参数
    START_DATE = "20250901"
    END_DATE = "20251001"
    INITIAL_CAPITAL = 200000.0
    POSITION_SIZE_PER_TRADE = 20000
    # 交易成本参数（示例配置，执行前设置）
    COMMISSION = 0.0005      # 0.05% 手续费
    MIN_COMMISSION = 5      # 最低 5 元
    SLIPPAGE = 0.001        # 0.1% 滑点
    SLIPPAGE_MODE = "both"  # 对开盘买入与收盘卖出均应用滑点
    
    # 创建并运行回测
    backtester = BacktestingRunner(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        position_size=POSITION_SIZE_PER_TRADE,
        commission=COMMISSION,
        min_commission=MIN_COMMISSION,
        slippage=SLIPPAGE,
        slippage_mode=SLIPPAGE_MODE
    )
    
    results = backtester.run()
    
    print("\n回测完成！")
    return results


if __name__ == "__main__":
    main()