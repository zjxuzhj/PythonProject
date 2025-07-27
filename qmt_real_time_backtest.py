import math
import os
from datetime import datetime, timedelta, time

import numpy as np
import pandas as pd
import xtquant.xtdata as xtdata

import first_limit_up_ma5_normal_scan as scan
import getAllStockCsv

query_tool = getAllStockCsv.StockQuery()

class Backtester:
    """
    一个用于运行特定股票交易策略回测的类。

    该类负责处理投资组合管理、交易执行和日志记录。
    """

    def __init__(self, start_date, end_date, initial_capital=200000.0, position_size=20000.0):
        self.start_dt = datetime.strptime(start_date, "%Y%m%d")
        self.end_dt = datetime.strptime(end_date, "%Y%m%d")
        self.initial_capital = initial_capital
        self.position_size = position_size

        # ---卖出策略参数 ---
        self.SELL_MA_BREAKDOWN_THRESHOLD = -0.003  # 跌破MA5下方千分之三
        self.MAX_HOLD_DAYS = 15  # 最大持有天数
        self.LIMIT_RATE = 0.10  # 涨跌停板比率，可根据需要调整(例如ST股为0.05)

        # 单一缓存文件设置
        self.cache_path = 'all_targets_cache.csv'
        self.load_scan_cache()

        # 投资组合状态
        self.cash = initial_capital
        self.positions = {} # value: {'shares': int, 'buy_price': float, 'buy_date': str, 'hold_days': int}
        self.portfolio_value = initial_capital
        self.trade_log = []

        # 交易时间限制
        self.BUY_CUTOFF_TIME = time(14, 30)  # 买入截止时间
        self.SELL_CHECK_TIME = time(14, 57)  # 卖出检查时间

    # --- 加载总缓存文件的方法 ---
    def load_scan_cache(self):
        """在回测开始时加载总的选股缓存文件到内存中。"""
        try:
            print(f"正在从 {self.cache_path} 加载选股缓存...")
            self.scan_cache_df = pd.read_csv(self.cache_path)
            # 确保date列是字符串类型，以便于比较
            self.scan_cache_df['date'] = self.scan_cache_df['date'].astype(str)
        except FileNotFoundError:
            print("缓存文件未找到，将创建一个新的。")
            self.scan_cache_df = pd.DataFrame(columns=['date', 'stock_code'])

    def get_daily_data_with_metrics(self, stock_code, start_date, end_date):
        """获取日线数据并计算所需指标(MA5, 涨跌停价)。"""
        df = xtdata.get_local_data(
            field_list=['time', 'open', 'high', 'low', 'close', 'volume'],  # 需要pre_close来计算涨跌停
            stock_list=[stock_code],
            period='1d',
            start_time=start_date,
            end_time=end_date,
            fill_data=True
        )
        if stock_code not in df or df[stock_code].empty:
            return pd.DataFrame()

        stock_df = df[stock_code].sort_index(ascending=True)
        stock_df['pre_close'] = stock_df['close'].shift(1)
        stock_df['ma5'] = stock_df['close'].rolling(window=5, min_periods=1).mean()
        # 计算理论涨跌停价
        stock_df['limit_price'] = (stock_df['pre_close'] * (1 + self.LIMIT_RATE)).round(2)
        stock_df['down_limit_price'] = (stock_df['pre_close'] * (1 - self.LIMIT_RATE)).round(2)
        return stock_df


    # --- 从单一缓存文件中获取/更新选股结果 ---
    def get_target_stocks_from_cache(self, date_str):
        """
        获取目标股票列表。优先从内存缓存查找，否则调用扫描函数并更新缓存文件和内存。
        """
        # 1. 在内存中查找该日期是否已被缓存
        cached_stocks_df = self.scan_cache_df[self.scan_cache_df['date'] == date_str]

        # 检查'stock_code'列是否存在且有值，处理扫描过但无结果的情况 (NaN)
        if not cached_stocks_df.empty:
            print(f"从内存缓存中找到 {date_str} 的选股结果。")
            # 如果扫描过但没有结果，stock_code列可能是NaN，需要过滤掉
            return cached_stocks_df['stock_code'].dropna().tolist()

        # 2. 如果内存中没有，则执行扫描
        print(f"缓存未找到 {date_str} 的数据，执行实时扫描...")
        target_stocks = scan.backtest_on_date(date_str)

        # 3. 准备新数据以便存入缓存
        if target_stocks:
            new_data_df = pd.DataFrame({
                'date': [date_str] * len(target_stocks),
                'stock_code': list(target_stocks)
            })
        else:
            # 如果当天没有选出任何股票，也记录下来，date有值，stock_code为空(NaN)
            # 这样下次就不会重复扫描了
            new_data_df = pd.DataFrame([{'date': date_str, 'stock_code': None}])

        # 4. 将新数据追加到CSV文件
        # 'a'模式为追加, header只在文件不存在或为空时写入
        should_write_header = not os.path.exists(self.cache_path) or os.path.getsize(self.cache_path) == 0
        new_data_df.to_csv(self.cache_path, mode='a', header=should_write_header, index=False)

        # 5. 更新内存中的DataFrame
        self.scan_cache_df = pd.concat([self.scan_cache_df, new_data_df], ignore_index=True)

        return target_stocks if target_stocks else []

    def get_daily_data(self, stock_code, start_date, end_date):
        """获取单个股票的日线历史数据。"""
        # 使用 get_market_data 获取日线数据通常更合适
        df = xtdata.get_local_data(
            field_list=['time', 'open', 'high', 'low', 'close', 'volume'],
            stock_list=[stock_code],
            period='1d',
            start_time=start_date,
            end_time=end_date,
            fill_data=True  # 确保对非交易日进行数据填充
        )
        if stock_code in df and not df[stock_code].empty:
            return df[stock_code].sort_index(ascending=True)
        return pd.DataFrame()

    def get_minute_data(self, stock_code, target_date):
        """获取单个股票在特定日期的1分钟K线数据。"""
        data = xtdata.get_local_data(
            field_list=['open', 'high', 'low', 'close'],
            stock_list=[stock_code],
            period='1m',
            start_time=target_date,
            end_time=target_date,
        )
        if stock_code in data and not data[stock_code].empty:
            df = data[stock_code]
            try:
                # 将数字索引转换为datetime对象
                df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d%H%M%S')
                return df
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def calculate_buy_target_price(self, stock_code, current_date_str):
        """
        基于前4天的MA5加上当天预测价格来计算目标买入价。
        """
        current_date = datetime.strptime(current_date_str, "%Y%m%d")
        # 获取当前日期之前的4个交易日的数据
        start_fetch_date = (current_date - timedelta(days=15)).strftime("%Y%m%d")
        end_fetch_date = (current_date - timedelta(days=1)).strftime("%Y%m%d")

        hist_df = self.get_daily_data(stock_code, start_fetch_date, end_fetch_date)

        if hist_df.empty or len(hist_df) < 4:
            print(f"  - [信息] {stock_code} 的历史数据不足，无法计算买入价。")
            return None

        # 取最后4天的数据
        last_4_days = hist_df.tail(4)
        if len(last_4_days) < 4:
            return None

        last_close = last_4_days.iloc[-1]['close']
        predicted_close = last_close * 1.04

        # 将历史收盘价与预测收盘价合并
        prices = last_4_days['close'].tolist()
        prices.append(predicted_close)

        # 计算5日均价
        target_price = round(sum(prices) / 5, 2)
        return target_price

    def log_trade(self, date_time, stock_code, action, shares, price, reason="", pnl=None):
        """将一笔交易记录到交易日志中。"""
        amount = shares * price
        log_entry = {
            "交易时间": date_time,
            "股票代码": stock_code,
            "操作": action,
            "股数": shares,
            "价格": price,
            "金额": amount,
            "原因": reason,
            "盈亏": pnl if pnl is not None else ""
        }
        self.trade_log.append(log_entry)
        trade_time_str = date_time.strftime('%H:%M')  # 提取时间，格式化为 HH:MM
        action_cn = "买入" if action == "BUY" else "卖出"
        reason_str = f" ({reason})" if reason else ""

        pnl_str = f" | 盈亏: {pnl:,.2f}" if pnl is not None else ""
        print(f"  -> {trade_time_str} {action_cn}: {query_tool.get_name_by_code(stock_code)} | {shares} 股 @ {price:.2f} | 金额: {amount:,.2f}{reason_str}{pnl_str}")

    def check_sell_conditions(self, stock_code, position, current_dt):
        """
        根据当前日期和持仓信息，检查所有卖出条件。
        返回 (卖出价格, 卖出原因) 或 (None, None)。
        """
        date_str = current_dt.strftime("%Y%m%d")

        # 获取包含当天和历史的日线数据以进行判断
        start_fetch_date = (current_dt - timedelta(days=20)).strftime("%Y%m%d")
        daily_df = self.get_daily_data_with_metrics(stock_code, start_fetch_date, date_str)

        # 需要至少两天数据（今天和昨天）
        if len(daily_df) < 2:
            return None, None

        today_data = daily_df.iloc[-1]
        prev_day_data = daily_df.iloc[-2]

        # --- 按优先级检查卖出条件 ---

        # if not np.isnan(today_data['ma5']) and today_data['ma5'] > 0:
        #     if today_data['close'] < today_data['ma5']:
        #          return today_data['close'], '跌破五日线'
        # # 条件1: 跌停止损 (前一天跌停)
        if prev_day_data['close'] <= prev_day_data['down_limit_price']:
            return today_data['close'], '跌停止损'

        # 条件2: 断板止盈 (前一天涨停，今天没涨停)
        if prev_day_data['close'] >= prev_day_data['limit_price']:
            if today_data['close'] < today_data['limit_price']:
                return today_data['close'], '断板止盈'
            else:
                return None, None  # 今天继续涨停，持有

        # 条件3: 炸板卖出 (当天触及涨停但未封板，且昨天未涨停)
        is_limit_touched = (today_data['high'] >= today_data['limit_price'])
        is_not_closed_at_limit = (today_data['close'] < today_data['limit_price'])
        is_prev_not_limit = (prev_day_data['close'] < prev_day_data['limit_price'])

        if is_limit_touched and is_not_closed_at_limit and is_prev_not_limit:
            return today_data['close'], '炸板卖出'

        # 条件4: 跌破五日线千分之三
        if not np.isnan(today_data['ma5']) and today_data['ma5'] > 0:
            if (today_data['close'] - today_data['ma5']) / today_data['ma5'] <= self.SELL_MA_BREAKDOWN_THRESHOLD:
                return today_data['close'], '跌破五日线'

        # 条件5: 最大持有天数限制
        if position['hold_days'] >= self.MAX_HOLD_DAYS:
            return today_data['close'], '持有超限'

        # 如果无任何卖出条件满足
        return None, None

    def run(self):
        current_dt = self.start_dt
        while current_dt <= self.end_dt:
            date_str = current_dt.strftime("%Y%m%d")
            print(f"\n--- 交易日: {current_dt.strftime('%Y-%m-%d')} ---")
            print(f"交易前现金: {self.cash:.2f}")
            if current_dt.weekday() >= 5:
                print("周末，跳过。")
                current_dt += timedelta(days=1)
                continue

            # 1. 首先处理当天的买入操作
            print("正在扫描新的可买入股票...")
            target_stocks = self.get_target_stocks_from_cache(date_str)
            if not target_stocks:
                print("今天没有发现目标股票。")
            else:
                potential_buys = []
                for stock_code in target_stocks:
                    if self.cash < self.position_size: break
                    buy_target_price = self.calculate_buy_target_price(stock_code, date_str)
                    if buy_target_price is None: continue
                    min_df = self.get_minute_data(stock_code, date_str)
                    if min_df.empty: continue
                    touched_df = min_df[(min_df['low'] <= buy_target_price) & (min_df['high'] >= buy_target_price)]
                    if not touched_df.empty:
                        first_touch_time = touched_df.index[0]
                        if first_touch_time.time() < self.BUY_CUTOFF_TIME:
                            potential_buys.append(
                                {'time': first_touch_time, 'stock': stock_code, 'price': buy_target_price})

                potential_buys.sort(key=lambda x: x['time'])
                print(f"在 {self.BUY_CUTOFF_TIME.strftime('%H:%M')} 前发现 {len(potential_buys)} 个潜在买入机会。")

                skipped_buys = []  # 初始化一个列表，用于存放错过的股票

                for buy_order in potential_buys:
                    if buy_order['stock'] in self.positions:
                        print(f"  - [跳过] 股票 {buy_order['stock']} 已在持仓中，本次不再买入。")
                        continue  # 跳过这笔交易，继续检查下一个
                    if self.cash >= self.position_size:
                        buy_price = buy_order['price']
                        shares_to_buy = math.floor((self.position_size / buy_price) / 100) * 100
                        if shares_to_buy > 0:
                            cost = shares_to_buy * buy_price
                            self.cash -= cost
                            self.positions[buy_order['stock']] = {
                                'shares': shares_to_buy,
                                'buy_price': buy_price,
                                'buy_date': date_str,
                                'hold_days': 0
                            }
                            self.log_trade(buy_order['time'], buy_order['stock'], "BUY", shares_to_buy, buy_price)
                        else:
                            print(f"  - [跳过] 资金不足以购买至少100股的 {buy_order['stock']} (价格: {buy_price:.2f})。")
                            skipped_buys.append(query_tool.get_name_by_code(buy_order['stock']))
                    else:
                        skipped_buys.append(query_tool.get_name_by_code(buy_order['stock']))

                # 循环结束后，检查错过的列表是否为空
                if skipped_buys:
                    print("今日现金不足，无法进行更多买入。")
                    print(f"  - 当日因现金不足错过的买入机会共 {len(skipped_buys)} 个。")
                    print(f"  - 错过股票: {', '.join(skipped_buys)}")

            # 2. 然后处理当天收盘后的卖出决策
            positions_to_check = list(self.positions.keys())
            if not positions_to_check:
                print("没有需要卖出的持仓。")
            else:
                print("正在检查持仓的卖出信号...")
                for stock_code in positions_to_check:
                    position = self.positions[stock_code]
                    if position['buy_date'] == date_str:
                        continue

                    position['hold_days'] += 1

                    sell_price, sell_reason = self.check_sell_conditions(stock_code, position, current_dt)

                    if sell_price is not None:
                        shares_to_sell = position['shares']
                        buy_price = position['buy_price']

                        # --- 这里是关键的修改：计算盈亏并传递 ---
                        profit_loss = (sell_price - buy_price) * shares_to_sell

                        self.cash += shares_to_sell * sell_price
                        self.log_trade(
                            date_time=current_dt,
                            stock_code=stock_code,
                            action="SELL",
                            shares=shares_to_sell,
                            price=sell_price,
                            reason=sell_reason,
                            pnl=profit_loss  # 传递盈亏额
                        )
                        del self.positions[stock_code]

            # 3. 最后进行每日总结
            holdings_value = 0
            daily_holdings_details = []  # 用于存储当日持仓详情

            for stock_code, position in self.positions.items():
                eod_df = self.get_daily_data_with_metrics(stock_code, date_str, date_str)
                last_close = eod_df.iloc[-1]['close'] if not eod_df.empty else position['buy_price']

                # 计算持仓详情
                shares = position['shares']
                buy_price = position['buy_price']
                market_value = shares * last_close
                pnl = (last_close - buy_price) * shares
                daily_holdings_details.append({
                    "code": stock_code, "shares": shares, "close": last_close,
                    "value": market_value, "pnl": pnl
                })
                holdings_value += market_value

            self.portfolio_value = self.cash + holdings_value
            print(
                f"交易日 {date_str} 结束: 现金: {self.cash:,.2f}, 持仓市值: {holdings_value:,.2f}, 投资组合总值: {self.portfolio_value:,.2f}")

            if daily_holdings_details:
                print("  当日持仓详情:")
                print("  " + "-" * 65)
                print(f"  {'股票名称':<6} {'持仓股数':>4} {'收盘价':>6} {'持仓市值':>8} {'浮动盈亏':>9}")
                for item in daily_holdings_details:
                    print(
                        f"  {query_tool.get_name_by_code(item['code']):<6} {item['shares']:>6,} {item['close']:>8.2f} {item['value']:>12,.2f} {item['pnl']:>10,.2f}")
                print("  " + "-" * 65)

            current_dt += timedelta(days=1)

        # 最终报告
        print("\n--- 回测结束 ---")
        print(f"初始资金: {self.initial_capital:,.2f}")
        print(f"最终投资组合价值: {self.portfolio_value:,.2f}")
        pnl = self.portfolio_value - self.initial_capital
        pnl_percent = (pnl / self.initial_capital) * 100
        print(f"总盈亏: {pnl:,.2f} ({pnl_percent:.2f}%)")
        if self.trade_log:
            pd.DataFrame(self.trade_log).to_excel("trading_log.xlsx", index=False)
            print("\n交易日志已保存到 'trading_log.xlsx'。")
        else:
            print("\n回测期间未执行任何交易。")


if __name__ == '__main__':
    # --- 配置区 ---
    # 注意：你必须提供一个包含有效本地数据的日期范围。
    # 示例日期仅用于演示。
    START_DATE = "20250501"
    END_DATE = "20250723"
    INITIAL_CAPITAL = 200000.0
    POSITION_SIZE_PER_TRADE = 21000.0

    # --- 运行回测 ---
    backtester = Backtester(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        position_size=POSITION_SIZE_PER_TRADE
    )
    backtester.run()
