"""
Backtrader 策略类
将原 qmt_real_time_backtest.py 中的 Backtester 类迁移到 Backtrader 框架
"""

import math
import pandas as pd
import backtrader as bt
from datetime import datetime, time, timedelta
from collections import defaultdict

import first_limit_up_ma5_normal_scan as scan
import getAllStockCsv
from common_sell_logic import get_sell_decision, MarketDataContext
from stock_info import StockInfo
from backtrader_data_feed import DataManager


class FirstLimitUpStrategy(bt.Strategy):
    """
    首板策略 - Backtrader 版本
    
    将原有的 Backtester 类逻辑迁移到 Backtrader 框架中
    """
    
    # 策略参数
    params = (
        ('position_size', 20000.0),  # 每笔交易金额
        ('limit_rate', 0.10),  # 涨跌停比率
        ('benchmark_ticker', '000300.SH'),  # 基准指数
        ('risk_free_rate', 0.02),  # 无风险利率
        ('buy_cutoff_time', time(14, 30)),  # 买入截止时间
        ('sell_check_time', time(14, 57)),  # 卖出检查时间
        ('cache_path', 'all_targets_cache_5.csv'),  # 缓存文件路径
    )
    
    def __init__(self):
        """初始化策略"""
        # 初始化查询工具
        self.query_tool = getAllStockCsv.StockQuery()
        self.data_manager = DataManager(self.params.limit_rate)
        
        # 加载选股缓存
        self.load_scan_cache()
        
        # 持仓管理
        self.positions_info = {}  # 存储持仓详细信息
        self.trade_log = []  # 交易日志
        self.daily_values = []  # 每日净值记录
        
        # 订单管理
        self.pending_orders = {}  # 待处理订单
        self.order_to_stock = {}  # 订单到股票的映射
        
        # 时间管理
        self.current_date = None
        self.trading_day_count = 0
        
        print(f"策略初始化完成，初始资金: {self.broker.getcash():.2f}")
    
    def load_scan_cache(self):
        """加载选股缓存文件"""
        try:
            print(f"正在从 {self.params.cache_path} 加载选股缓存...")
            self.scan_cache_df = pd.read_csv(self.params.cache_path)
            self.scan_cache_df['date'] = self.scan_cache_df['date'].astype(str)
            print(f"缓存加载完成，共 {len(self.scan_cache_df)} 条记录")
        except FileNotFoundError:
            print("缓存文件未找到，将创建一个新的。")
            self.scan_cache_df = pd.DataFrame(columns=['date', 'stock_code'])
    
    def next(self):
        """策略主逻辑 - 每个交易日调用"""
        # 获取当前日期
        current_dt = self.datas[0].datetime.date(0)
        date_str = current_dt.strftime("%Y%m%d")
        
        # 如果是新的交易日
        if self.current_date != current_dt:
            self.current_date = current_dt
            self.trading_day_count += 1
            
            print(f"\n--- 交易日: {current_dt} ---")
            print(f"交易前现金: {self.broker.getcash():.2f}")
            
            # 更新持仓天数
            self._update_position_days()
            
            # 执行交易逻辑
            self._process_sell_signals()
            self._process_buy_signals(date_str)
            
            # 记录每日净值
            self._record_daily_value(current_dt)
    
    def _update_position_days(self):
        """更新持仓天数"""
        for stock_code in list(self.positions_info.keys()):
            if stock_code in self.positions_info:
                self.positions_info[stock_code]['hold_days'] += 1
    
    def _process_sell_signals(self):
        """处理卖出信号"""
        if not self.positions_info:
            print("没有需要卖出的持仓。")
            return
            
        print("正在检查持仓的卖出信号...")
        
        for stock_code in list(self.positions_info.keys()):
            position_info = self.positions_info[stock_code]
            
            # 跳过当日买入的股票
            if position_info['hold_days'] == 0:
                continue
                
            # 检查卖出条件
            should_sell, sell_reason = self._check_sell_conditions(stock_code, position_info)
            
            if should_sell:
                # 执行卖出
                self._execute_sell_order(stock_code, sell_reason)
    
    def _check_sell_conditions(self, stock_code, position_info):
        """
        检查卖出条件
        
        Args:
            stock_code: 股票代码
            position_info: 持仓信息
            
        Returns:
            tuple: (是否卖出, 卖出原因)
        """
        try:
            current_dt = self.current_date
            date_str = current_dt.strftime("%Y%m%d")
            
            # 获取历史数据用于判断
            start_fetch_date = (current_dt - timedelta(days=30)).strftime("%Y%m%d")
            daily_df = self.data_manager.get_stock_data(stock_code, start_fetch_date, date_str)
            
            if len(daily_df) < 2:
                return False, None
                
            today_data = daily_df.iloc[-1]
            prev_day_data = daily_df.iloc[-2]
            
            # 构建市场数据上下文
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
                name=self.query_tool.get_name_by_code(stock_code),
                market_value=self.query_tool.get_stock_market_value(stock_code),
                theme=self.query_tool.get_theme_by_code(stock_code),
            )
            
            # 调用卖出决策逻辑
            should_sell, reason = get_sell_decision(stock_info, position_info, market_data)
            
            return should_sell, reason
            
        except Exception as e:
            print(f"检查股票 {stock_code} 卖出条件时出错: {e}")
            return False, None
    
    def _execute_sell_order(self, stock_code, reason):
        """
        执行卖出订单
        
        Args:
            stock_code: 股票代码
            reason: 卖出原因
        """
        try:
            position_info = self.positions_info[stock_code]
            
            # 获取当前价格（收盘价）
            current_dt = self.current_date
            date_str = current_dt.strftime("%Y%m%d")
            daily_df = self.data_manager.get_stock_data(stock_code, date_str, date_str)
            
            if daily_df.empty:
                print(f"无法获取股票 {stock_code} 当日价格数据")
                return
                
            sell_price = daily_df.iloc[-1]['close']
            
            # 计算盈亏
            buy_price = position_info['buy_price']
            shares = position_info['shares']
            profit_loss = (sell_price - buy_price) * shares
            
            # 执行 Backtrader 卖出订单
            order = self.sell(size=shares)
            
            # 记录交易日志
            self._log_trade(
                date_time=current_dt,
                stock_code=stock_code,
                action="SELL",
                shares=shares,
                price=sell_price,
                reason=reason,
                pnl=profit_loss,
                buy_time_category=position_info.get('buy_time_category', 'Unknown')
            )
            
            # 清除持仓信息
            del self.positions_info[stock_code]
            
        except Exception as e:
            print(f"执行股票 {stock_code} 卖出订单时出错: {e}")
    
    def _process_buy_signals(self, date_str):
        """处理买入信号"""
        print("正在扫描新的可买入股票...")
        
        # 获取目标股票列表
        target_stocks = self._get_target_stocks_from_cache(date_str)
        
        if not target_stocks:
            print("今天没有发现目标股票。")
            return
            
        # 处理买入逻辑
        potential_buys = []
        
        for stock_code in target_stocks:
            if self.broker.getcash() < self.params.position_size:
                break
                
            # 检查是否已持仓
            if stock_code in self.positions_info:
                print(f"  - [跳过] 股票 {stock_code} 已在持仓中，本次不再买入。")
                continue
                
            # 计算买入目标价格
            buy_target_price = self._calculate_buy_target_price(stock_code, date_str)
            if buy_target_price is None:
                continue
                
            # 检查分钟数据中是否触及目标价格
            min_df = self.data_manager.get_minute_data(stock_code, date_str)
            if min_df.empty:
                continue
                
            # 查找触及目标价格的时间点
            touched_df = min_df[
                (min_df['low'] <= buy_target_price) & 
                (min_df['high'] >= buy_target_price)
            ]
            
            if not touched_df.empty:
                first_touch_time = touched_df.index[0]
                
                # 防御性检查：确保 first_touch_time 有 .time() 方法
                try:
                    if hasattr(first_touch_time, 'time'):
                        touch_time = first_touch_time.time()
                    elif hasattr(first_touch_time, 'to_pydatetime'):
                        # 如果是 Timestamp，转换为 datetime 再获取时间
                        touch_time = first_touch_time.to_pydatetime().time()
                    else:
                        # 如果是字符串，尝试解析
                        if isinstance(first_touch_time, str):
                            from datetime import datetime
                            dt = datetime.strptime(first_touch_time, '%Y-%m-%d %H:%M:%S')
                            touch_time = dt.time()
                        else:
                            print(f"警告: first_touch_time 类型不支持: {type(first_touch_time)}, 值: {first_touch_time}")
                            continue
                    
                    if touch_time < self.params.buy_cutoff_time:
                        potential_buys.append({
                            'time': first_touch_time,
                            'stock': stock_code,
                            'price': buy_target_price
                        })
                        
                except Exception as e:
                    print(f"处理 first_touch_time 时出错 (股票: {stock_code}): {e}")
                    print(f"first_touch_time 类型: {type(first_touch_time)}, 值: {first_touch_time}")
                    continue
        
        # 按时间排序买入机会
        potential_buys.sort(key=lambda x: x['time'])
        print(f"在 {self.params.buy_cutoff_time.strftime('%H:%M')} 前发现 {len(potential_buys)} 个潜在买入机会。")
        
        # 执行买入订单
        skipped_buys = []
        
        for buy_order in potential_buys:
            if self.broker.getcash() >= self.params.position_size:
                success = self._execute_buy_order(buy_order)
                if not success:
                    skipped_buys.append(
                        self.query_tool.get_name_by_code(buy_order['stock'])
                    )
            else:
                skipped_buys.append(
                    self.query_tool.get_name_by_code(buy_order['stock'])
                )
        
        # 报告错过的买入机会
        if skipped_buys:
            print("今日现金不足，无法进行更多买入。")
            print(f"  - 当日因现金不足错过的买入机会共 {len(skipped_buys)} 个。")
            print(f"  - 错过股票: {', '.join(skipped_buys)}")
    
    def _execute_buy_order(self, buy_order):
        """
        执行买入订单
        
        Args:
            buy_order: 买入订单信息
            
        Returns:
            bool: 是否成功执行
        """
        try:
            stock_code = buy_order['stock']
            buy_price = buy_order['price']
            buy_time = buy_order['time']
            
            # 计算买入股数（100股的整数倍）
            shares_to_buy = math.floor((self.params.position_size / buy_price) / 100) * 100
            
            if shares_to_buy <= 0:
                print(f"  - [跳过] 资金不足以购买至少100股的 {stock_code} (价格: {buy_price:.2f})。")
                return False
            
            # 执行 Backtrader 买入订单
            order = self.buy(size=shares_to_buy)
            
            # 记录持仓信息
            buy_time_category = self._get_buy_time_category(buy_time)
            self.positions_info[stock_code] = {
                'shares': shares_to_buy,
                'buy_price': buy_price,
                'buy_date': self.current_date.strftime("%Y%m%d"),
                'hold_days': 0,
                'buy_time_category': buy_time_category
            }
            
            # 记录交易日志
            self._log_trade(
                date_time=buy_time,
                stock_code=stock_code,
                action="BUY",
                shares=shares_to_buy,
                price=buy_price,
                buy_time_category=buy_time_category
            )
            
            return True
            
        except Exception as e:
            print(f"执行买入订单时出错: {e}")
            return False
    
    def _get_target_stocks_from_cache(self, date_str):
        """从缓存获取目标股票列表"""
        try:
            # 从内存缓存查找
            cached_stocks_df = self.scan_cache_df[self.scan_cache_df['date'] == date_str]
            
            if not cached_stocks_df.empty:
                print(f"从内存缓存中找到 {date_str} 的选股结果。")
                return cached_stocks_df['stock_code'].dropna().tolist()
            
            # 如果缓存中没有，执行实时扫描
            print(f"缓存未找到 {date_str} 的数据，执行实时扫描...")
            target_stocks, _ = scan.get_target_stocks(target_date=date_str)
            
            # 更新缓存
            if target_stocks:
                new_records = pd.DataFrame([
                    {'date': date_str, 'stock_code': stock} 
                    for stock in target_stocks
                ])
                self.scan_cache_df = pd.concat([self.scan_cache_df, new_records], ignore_index=True)
            else:
                # 记录扫描过但无结果的日期
                new_record = pd.DataFrame([{'date': date_str, 'stock_code': None}])
                self.scan_cache_df = pd.concat([self.scan_cache_df, new_record], ignore_index=True)
            
            # 保存缓存文件
            self.scan_cache_df.to_csv(self.params.cache_path, index=False)
            
            return target_stocks
            
        except Exception as e:
            print(f"获取目标股票列表时出错: {e}")
            return []
    
    def _calculate_buy_target_price(self, stock_code, current_date_str):
        """计算买入目标价格"""
        try:
            # 获取前一日数据计算MA5
            prev_date = (datetime.strptime(current_date_str, "%Y%m%d") - timedelta(days=5)).strftime("%Y%m%d")
            df = self.data_manager.get_stock_data(stock_code, prev_date, current_date_str)
            
            if len(df) < 2:
                return None
                
            # 使用前一日的MA5作为买入目标价格
            ma5_price = df.iloc[-2]['ma5']  # 前一日的MA5
            
            return ma5_price if pd.notna(ma5_price) else None
            
        except Exception as e:
            print(f"计算股票 {stock_code} 买入目标价格时出错: {e}")
            return None
    
    def _get_buy_time_category(self, buy_datetime):
        """获取买入时间分类"""
        buy_time = buy_datetime.time()
        
        if time(9, 30) <= buy_time < time(10, 0):
            return "09:30-10:00"
        elif time(10, 0) <= buy_time < time(11, 30):
            return "10:00-11:30"
        elif time(13, 0) <= buy_time < time(14, 50):
            return "13:00-14:50"
        else:
            return "Other"
    
    def _log_trade(self, date_time, stock_code, action, shares, price, reason="", pnl=None, buy_time_category=None):
        """记录交易日志"""
        amount = shares * price
        log_entry = {
            "交易时间": date_time,
            "股票代码": stock_code,
            "操作": action,
            "股数": shares,
            "价格": price,
            "金额": amount,
            "原因": reason,
            "盈亏": pnl if pnl is not None else "",
            "买入时段": buy_time_category
        }
        self.trade_log.append(log_entry)
        
        # 打印交易信息
        if isinstance(date_time, datetime):
            trade_time_str = date_time.strftime('%H:%M')
        else:
            trade_time_str = str(date_time)
            
        action_cn = "买入" if action == "BUY" else "卖出"
        reason_str = f" ({reason})" if reason else ""
        pnl_str = f" | 盈亏: {pnl:,.2f}" if pnl is not None else ""
        
        print(f"  -> {trade_time_str} {action_cn}: {self.query_tool.get_name_by_code(stock_code)} | "
              f"{shares} 股 @ {price:.2f} | 金额: {amount:,.2f}{reason_str}{pnl_str}")
    
    def _record_daily_value(self, current_dt):
        """记录每日投资组合价值"""
        portfolio_value = self.broker.getvalue()
        self.daily_values.append((current_dt, portfolio_value))
        
        # 计算持仓详情
        holdings_value = portfolio_value - self.broker.getcash()
        
        print(f"交易日 {current_dt.strftime('%Y%m%d')} 结束: "
              f"现金: {self.broker.getcash():,.2f}, "
              f"持仓市值: {holdings_value:,.2f}, "
              f"投资组合总值: {portfolio_value:,.2f}")
        
        # 打印持仓详情
        if self.positions_info:
            print("  当日持仓详情:")
            print("  " + "-" * 65)
            print(f"  {'股票名称':<6} {'持仓股数':>4} {'买入价':>6} {'持仓天数':>6}")
            
            for stock_code, position in self.positions_info.items():
                print(f"  {self.query_tool.get_name_by_code(stock_code):<6} "
                      f"{position['shares']:>6,} "
                      f"{position['buy_price']:>8.2f} "
                      f"{position['hold_days']:>8}")
            print("  " + "-" * 65)
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"买入订单执行完成: 价格 {order.executed.price:.2f}, "
                      f"数量 {order.executed.size}, "
                      f"手续费 {order.executed.comm:.2f}")
            else:
                print(f"卖出订单执行完成: 价格 {order.executed.price:.2f}, "
                      f"数量 {order.executed.size}, "
                      f"手续费 {order.executed.comm:.2f}")
                      
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"订单被取消/拒绝: {order.status}")
    
    def stop(self):
        """策略结束时调用"""
        print("\n--- 回测结束 ---")
        
        initial_capital = self.broker.startingcash
        final_value = self.broker.getvalue()
        
        print(f"初始资金: {initial_capital:,.2f}")
        print(f"最终投资组合价值: {final_value:,.2f}")
        
        pnl = final_value - initial_capital
        pnl_percent = (pnl / initial_capital) * 100
        print(f"总盈亏: {pnl:,.2f} ({pnl_percent:.2f}%)")
        
        # 保存交易日志
        if self.trade_log:
            df = pd.DataFrame(self.trade_log)
            df.to_excel("backtrader_trading_log.xlsx", index=False)
            print("\n交易日志已保存到 'backtrader_trading_log.xlsx'。")
        else:
            print("\n回测期间未执行任何交易。")