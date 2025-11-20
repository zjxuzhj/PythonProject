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
                 slippage_mode: str = "both",
                 max_positions: Optional[int] = None,
                 use_dynamic_allocation: Optional[bool] = None):
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
        
        # 参数验证
        if initial_capital is None or float(initial_capital) <= 0:
            raise ValueError("初始资金必须为正数")
        if use_dynamic_allocation:
            if max_positions is None or int(max_positions) <= 0:
                raise ValueError("最大持仓数量必须为正数")

        # 交易状态
        self.cash = initial_capital
        self.positions = {}  # {stock_code: {shares, buy_price, buy_date, hold_days, buy_time_category}}
        self.trade_log = []
        self.daily_values = []
        # 动态仓位管理参数（兼容旧参数）
        self.use_dynamic_allocation = True if use_dynamic_allocation is None else bool(use_dynamic_allocation)
        if self.use_dynamic_allocation:
            # 若提供max_positions，则用 1/max_positions 作为目标比例；否则回退到由position_size推导比例
            if max_positions and int(max_positions) > 0:
                self.max_slots = int(max_positions)
                self.dynamic_ratio = 1.0 / float(self.max_slots)
            else:
                # 兼容旧参数：按标准仓位=position_size 推导近似比例
                self.dynamic_ratio = min(1.0, max(0.01, float(position_size) / float(initial_capital)))
                self.max_slots = max(1, int(round(1.0 / self.dynamic_ratio)))
        else:
            self.dynamic_ratio = 0.0
            self.max_slots = 1
        self.lot_size = 100                 # 交易最小单位（A股100股）
        self.rebalancing_log: List[Dict] = []
        
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

    def _estimate_current_price(self, stock_code: str, date_str: str) -> float:
        """估算当日用于调仓的价格：优先当日开盘，其次当日收盘，其次买入价格。"""
        try:
            day_df = self.get_daily_data(stock_code, date_str, date_str)
            if not day_df.empty:
                if 'open' in day_df.columns and not pd.isna(day_df.iloc[-1]['open']):
                    return float(day_df.iloc[-1]['open'])
                if 'close' in day_df.columns and not pd.isna(day_df.iloc[-1]['close']):
                    return float(day_df.iloc[-1]['close'])
        except Exception:
            pass
        pos = self.positions.get(stock_code)
        return float(pos['buy_price']) if pos else 0.0

    def _estimate_total_asset(self, date_str: str) -> float:
        """估算当前总资产（现金 + 持仓按当日开盘/收盘估值）。"""
        total = float(self.cash)
        for code, pos in self.positions.items():
            px = self._estimate_current_price(code, date_str)
            if px > 0:
                total += pos['shares'] * px
            else:
                total += pos['shares'] * pos['buy_price']
        return float(total)

    def _record_rebalance(self, date_time: datetime, stock_code: str, action: str, target_value: float,
                           current_value: float, shares: int, price: float, resulting_ratio: float):
        """记录调仓日志（比例管理）。"""
        self.rebalancing_log.append({
            'datetime': date_time,
            'stock_code': stock_code,
            'action': action,
            'target_value': round(target_value, 2),
            'current_value': round(current_value, 2),
            'shares': int(shares),
            'price': round(price, 4),
            'resulting_ratio': round(resulting_ratio, 6)
        })
    
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
        if action == "BUY":
            # 详细买入日志格式：
            # -> HH:MM 买入: [股票名称] | [数量] 股 @ [价格] | 金额: [金额] | 价格类型: [类型] | 参考买入价: [参考价]
            try:
                time_str = date_time.strftime("%H:%M")
            except Exception:
                time_str = ""
            name = trade_record['stock_name']
            amount = float(shares) * float(price)
            price_type_str = str(trade_record.get('price_type', '') or '')
            ref_val = trade_record.get('ref_price', '')
            ref_str = ""
            try:
                if isinstance(ref_val, (int, float)):
                    ref_str = f"{float(ref_val):.2f}"
                elif isinstance(ref_val, str) and ref_val.strip() != "":
                    ref_str = f"{float(ref_val):.2f}"
            except Exception:
                ref_str = ""
            print(
                f"-> {time_str} 买入: {name} | {shares} 股 @ {price:.2f} | 金额: {amount:,.2f} | 价格类型: {price_type_str} | 参考买入价: {ref_str}"
            )
        else:
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
        # print(f"买入次数: {len(buy_trades)}")
        # print(f"卖出次数: {len(sell_trades)}")
        
        if len(sell_trades) > 0:
            # 计算盈亏统计
            total_pnl = sell_trades['pnl'].sum()
            win_trades = sell_trades[sell_trades['pnl'] > 0]
            lose_trades = sell_trades[sell_trades['pnl'] <= 0]
            
            win_rate = len(win_trades) / len(sell_trades) * 100
            avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
            avg_loss = lose_trades['pnl'].mean() if len(lose_trades) > 0 else 0
            if abs(avg_loss) > 0:
                profit_loss_ratio = avg_win / abs(avg_loss)
            else:
                profit_loss_ratio = np.inf
            print(f"总盈亏: {total_pnl:.2f}，胜率: {win_rate:.2f}%，盈亏比: {'无穷大' if profit_loss_ratio == np.inf else f'{profit_loss_ratio:.2f}:1'}")
            # print(f"平均盈利: {avg_win:.2f}")
            # print(f"平均亏损: {avg_loss:.2f}")

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
            calmar_ratio = annual_return / abs(max_drawdown)

            print(f"\n=== 收益与风险指标 ===")
            print(f"总收益率: {total_return:.2%}，最大回撤: {max_drawdown:.2%},年化收益率: {annual_return:.2%}，卡玛率:{calmar_ratio:.2f}，年化波动率: {annual_volatility:.2%}")
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
        # 记录本次运行开始时间（精确到秒），用于图片命名
        try:
            self.run_started_at = datetime.now()
        except Exception:
            # 兜底：若无法获取当前时间，则使用开始日期的 00:00:00
            try:
                self.run_started_at = datetime.strptime(self.start_date + "_000000", "%Y%m%d_%H%M%S")
            except Exception:
                self.run_started_at = datetime.now()
        # 初始化图表计数器，用于多图按时间顺序编号
        try:
            self.plot_counter
        except AttributeError:
            self.plot_counter = 0
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
                                total_asset_now = self._estimate_total_asset(date_str) if self.use_dynamic_allocation else self.initial_capital
                                target_value = total_asset_now * self.dynamic_ratio if self.use_dynamic_allocation else self.position_size
                                planned_shares = math.floor((target_value / buy_target_price) / self.lot_size) * self.lot_size
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
                # 限制当日最多参与的入选股票数量（按目标占比10%合计不超过100%）
                selected_universe = [o['stock'] for o in potential_buys]
                if self.use_dynamic_allocation:
                    selected_universe = selected_universe[: self.max_slots]
                for buy_order in potential_buys:
                    # 超过当日可分配上限则跳过
                    if self.use_dynamic_allocation and buy_order['stock'] not in selected_universe:
                        continue
                    if buy_order['stock'] in self.positions:
                        print(f"  - [跳过] 股票 {buy_order['stock']} 已在持仓中，本次不再买入。")
                        continue
                    
                    # 动态目标仓位计算
                    total_asset_now = self._estimate_total_asset(date_str) if self.use_dynamic_allocation else (self.cash + 0)
                    raw_price = buy_order['price']
                    price_type = buy_order.get('price_type', '预估买入价')
                    ref_price = buy_order.get('ref_price', raw_price)
                    buy_price = self._apply_slippage(raw_price, action="BUY", price_type=price_type)

                    target_value = total_asset_now * self.dynamic_ratio if self.use_dynamic_allocation else self.position_size
                    current_value = 0.0
                    need_value = max(target_value - current_value, 0.0)
                    shares_to_buy = 0
                    if self.use_dynamic_allocation:
                        if need_value > 0 and buy_price > 0:
                            shares_to_buy = math.floor((need_value / buy_price) / self.lot_size) * self.lot_size
                    else:
                        shares_to_buy = math.floor((self.position_size / buy_price) / self.lot_size) * self.lot_size

                    # 现金检查与调整
                    can_buy = False
                    fee_est = self._calc_commission(max(0.0, shares_to_buy * buy_price))
                    cost_est = shares_to_buy * buy_price + fee_est
                    if shares_to_buy > 0 and self.cash >= cost_est:
                        can_buy = True
                    else:
                        # 资金不足时按可用现金缩减至可买数量
                        if self.use_dynamic_allocation and buy_price > 0 and (self.cash - self.min_commission) > buy_price:
                            max_shares = math.floor(((self.cash - self.min_commission) / buy_price) / self.lot_size) * self.lot_size
                            if max_shares > 0:
                                shares_to_buy = max_shares
                                fee_est = self._calc_commission(shares_to_buy * buy_price)
                                cost_est = shares_to_buy * buy_price + fee_est
                                can_buy = self.cash >= cost_est

                    if can_buy:
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
                        resulting_ratio = (shares_to_buy * buy_price) / max(1e-9, total_asset_now)
                        self._record_rebalance(buy_order['time'], buy_order['stock'], "BUY_ALLOC",
                                               target_value, current_value, shares_to_buy, buy_price, resulting_ratio)
                    else:
                        print(f"  - [跳过] 资金不足以购买至少{self.lot_size}股的 {buy_order['stock']} (价格: {buy_price:.2f})。")
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
            
            # 2b. 动态调仓（收盘价按比例平衡到目标占比=1/max_positions）
            if self.use_dynamic_allocation:
                    total_asset_close = self._estimate_total_asset(date_str)
                    for code in list(self.positions.keys()):
                        eod_df = self.get_daily_data_with_metrics(code, date_str, date_str)
                        close_px = float(eod_df.iloc[-1]['close']) if not eod_df.empty else self.positions[code]['buy_price']
                        current_val = self.positions[code]['shares'] * close_px
                        target_val = total_asset_close * self.dynamic_ratio
                        if current_val > target_val:
                            delta_val = current_val - target_val
                            sell_shares = math.floor((delta_val / close_px) / self.lot_size) * self.lot_size
                            if sell_shares > 0:
                                exec_sell_price = self._apply_slippage(close_px, action="SELL", price_type="收盘价")
                                gross_proceeds = sell_shares * exec_sell_price
                                sell_fee = self._calc_commission(gross_proceeds)
                                net_proceeds = gross_proceeds - sell_fee
                                self.cash += net_proceeds
                                self.positions[code]['shares'] -= sell_shares
                                if self.positions[code]['shares'] <= 0:
                                    del self.positions[code]
                                self._record_rebalance(current_dt, code, "REBAL_SELL", target_val, current_val,
                                                       sell_shares, exec_sell_price, target_val / max(1e-9, total_asset_close))
                        elif current_val < target_val:
                            delta_val = target_val - current_val
                            buy_shares = math.floor((delta_val / close_px) / self.lot_size) * self.lot_size
                            if buy_shares > 0:
                                exec_buy_price = self._apply_slippage(close_px, action="BUY", price_type="收盘价")
                                gross_cost = buy_shares * exec_buy_price
                                buy_fee = self._calc_commission(gross_cost)
                                total_cost = gross_cost + buy_fee
                                if self.cash >= total_cost:
                                    self.cash -= total_cost
                                    self.positions[code]['shares'] += buy_shares
                                    self._record_rebalance(current_dt, code, "REBAL_BUY", target_val, current_val,
                                                           buy_shares, exec_buy_price, target_val / max(1e-9, total_asset_close))
                
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
            
            # 保存调仓日志
            if hasattr(self, 'rebalancing_log') and self.rebalancing_log:
                reb_df = pd.DataFrame(self.rebalancing_log)
                reb_df.to_csv('rebalancing_log.csv', index=False, encoding='utf-8')
                print(f"调仓日志已保存到 rebalancing_log.csv")
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")

    def _configure_chinese_font(self):
        """配置 Matplotlib 中文字体，避免中文字符显示为方框或缺失。
        在 Windows 优先使用 'Microsoft YaHei' 或 'SimHei'；否则尝试常见 CJK 字体。
        """
        try:
            import os
            import matplotlib
            from matplotlib import font_manager as fm

            candidates = [
                ("Microsoft YaHei", r"C:\\Windows\\Fonts\\msyh.ttc"),
                ("SimHei", r"C:\\Windows\\Fonts\\simhei.ttf"),
                ("Arial Unicode MS", r"C:\\Windows\\Fonts\\ARIALUNI.TTF"),
                ("DengXian", r"C:\\Windows\\Fonts\\Deng.ttf"),
                ("Noto Sans CJK SC", None),
                ("Source Han Sans SC", None),
                ("PingFang SC", None)
            ]

            selected_name = None
            selected_path = None

            # 先尝试通过字体名称获取系统字体路径
            for name, fallback_path in candidates:
                try:
                    font_path = fm.findfont(name, fallback_to_default=False)
                    if font_path and os.path.exists(font_path):
                        selected_name = name
                        selected_path = font_path
                        break
                except Exception:
                    pass
                # 如果通过名称失败，尝试常见路径
                if fallback_path and os.path.exists(fallback_path):
                    selected_name = name
                    selected_path = fallback_path
                    break

            if selected_name and selected_path:
                try:
                    fm.fontManager.addfont(selected_path)
                except Exception:
                    pass
                matplotlib.rcParams['font.family'] = 'sans-serif'
                matplotlib.rcParams['font.sans-serif'] = [selected_name]
            else:
                # 最后兜底：设置一组常见的中文字体名称，若系统存在会自动匹配
                matplotlib.rcParams['font.family'] = 'sans-serif'
                matplotlib.rcParams['font.sans-serif'] = [
                    'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'Arial Unicode MS'
                ]

            # 修复负号显示为方块的问题
            matplotlib.rcParams['axes.unicode_minus'] = False
        except Exception:
            # 保持默认字体设置，不影响作图流程
            pass

    def plot_results(self, output_path: Optional[str] = None, dpi: int = 300) -> Optional[str]:
        """生成策略与基准的收益对比图。
        - 规格：16:9（1920x1080 像素），网格线、图例、标题与参数摘要。
        - 曲线：策略收益（#1f77b4）、沪深300（#d62728），线宽 2.5。
        - 轴：X 为日期自动优化并旋转 30°；Y 为百分比，主刻度 10%。
        - 输出：保存为 PNG 至 results/ 目录。
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.ticker import PercentFormatter, MultipleLocator
        except Exception as e:
            print(f"绘图失败（matplotlib 不可用）：{e}")
            return None

        # 在绘图前配置中文字体
        self._configure_chinese_font()

        if not self.daily_values:
            print("无每日净值数据，跳过绘图。")
            return None

        # 策略曲线
        daily_df = pd.DataFrame(self.daily_values).copy()
        try:
            daily_df['date'] = pd.to_datetime(daily_df['date'], format='%Y%m%d')
        except Exception:
            daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df = daily_df.sort_values('date').set_index('date')
        strat_returns = daily_df['total_value'] / float(self.initial_capital) - 1.0

        # 基准曲线
        bench_df = self.get_daily_data(self.BENCHMARK_TICKER, self.start_date, self.end_date)
        bench_series = None
        if isinstance(bench_df, pd.DataFrame) and not bench_df.empty:
            bench_df = bench_df.sort_index()
            try:
                bench_series = bench_df['close'] / bench_df['close'].iloc[0] - 1.0
            except Exception:
                bench_series = None
        else:
            print("基准数据为空，基准曲线不显示。")

        # 指标摘要
        try:
            total_return = (daily_df['total_value'].iloc[-1] - self.initial_capital) / self.initial_capital
            days = len(daily_df)
            annual_return = (1 + total_return) ** (252 / max(1, days)) - 1
            cummax = daily_df['total_value'].cummax()
            drawdown = (daily_df['total_value'] - cummax) / cummax
            max_drawdown = drawdown.min()
        except Exception:
            annual_return = 0.0
            max_drawdown = 0.0

        # 尺寸：1920x1080 at 300dpi => 6.4x3.6 inches
        fig_w_in, fig_h_in = 1920/300, 1080/300
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)

        ax.plot(strat_returns.index, strat_returns, color="#1f77b4", linewidth=2.5, label="策略收益")
        if bench_series is not None:
            # 对齐到策略日期索引
            aligned_bench = bench_series.reindex(strat_returns.index).ffill()
            ax.plot(aligned_bench.index, aligned_bench, color="#d62728", linewidth=2.5, label="沪深300")

        ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
        ax.legend(loc='best')
        ax.set_title("策略回测表现对比")
        ax.set_ylabel("收益率")
        ax.yaxis.set_major_locator(MultipleLocator(0.10))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate(rotation=30)

        summary = (
            f"起始日期: {self.start_date}\n"
            f"结束日期: {self.end_date}\n"
            f"年化收益率: {annual_return:.2%}\n"
            f"最大回撤: {max_drawdown:.2%}"
        )
        fig.text(0.98, 0.02, summary, ha='right', va='bottom', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

        try:
            # 统一命名：${开始时间}_${结束时间}_${运行开始时间}.${图片格式}
            # 要求：开始/结束时间仅保留日期（YYYYMMDD），运行开始时间仅保留当天时间 HHMMSS
            start_day = str(self.start_date).replace('-', '')[:8]
            end_day = str(self.end_date).replace('-', '')[:8]
            run_ts = None
            try:
                run_ts = self.run_started_at.strftime("%H%M%S")
            except Exception:
                run_ts = datetime.now().strftime("%H%M%S")

            # 解析输出格式与目录
            fmt = 'png'
            out_dir = os.path.join('results')
            if output_path:
                # 若传入文件路径，使用其扩展名；若为目录，使用该目录
                base, ext = os.path.splitext(output_path)
                if ext:
                    fmt = ext.lstrip('.').lower()
                    # 如果是文件路径，仍按规范重命名并输出到其所在目录
                    out_dir = os.path.dirname(output_path) or out_dir
                else:
                    # 作为目录使用
                    out_dir = output_path
            os.makedirs(out_dir, exist_ok=True)

            # 多图编号（按时间顺序），从 1 开始
            try:
                self.plot_counter += 1
            except Exception:
                self.plot_counter = 1

            seq_suffix = f"_{self.plot_counter:02d}" if self.plot_counter > 1 else ""
            file_name = f"{start_day}_{end_day}_{run_ts}{seq_suffix}.{fmt}"
            out_file = os.path.join(out_dir, file_name)

            plt.savefig(out_file, format=fmt, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"图表已保存: {out_file}")
            return out_file
        except Exception as e:
            print(f"保存图表失败: {e}")
            try:
                plt.close(fig)
            except Exception:
                pass
            return None


def main():
    """主函数"""
    # 是否生成图表由代码变量控制，不使用命令行参数
    GENERATE_PLOT = True
    # 回测参数
    START_DATE = "20250810"
    END_DATE = "20251030"
    INITIAL_CAPITAL = 200000.0  # 总资产金额
    MAX_POSITIONS = 11   # 最大持仓数量
    POSITION_SIZE_PER_TRADE = 20000  # 保留旧参数兼容（未启用比例制时使用）
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
        slippage_mode=SLIPPAGE_MODE,
        max_positions=MAX_POSITIONS,
        use_dynamic_allocation=True
    )
    
    results = backtester.run()
    
    print("\n回测完成！")

    if GENERATE_PLOT:
        backtester.plot_results()
    return results


if __name__ == "__main__":
    main()