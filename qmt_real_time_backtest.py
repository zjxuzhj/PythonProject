import xtquant.xtdata as xtdata
from datetime import datetime, timedelta, time
import pandas as pd
import os
# 假设你的扫描函数在这个文件中。
# 如果在不同文件中，请确保导入正确。
import first_limit_up_ma5_normal_scan as scan
import math


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

        # 单一缓存文件设置
        self.cache_path = 'all_targets_cache.csv'
        self.load_scan_cache()

        # 投资组合状态
        self.cash = initial_capital
        self.positions = {}  # 键: stock_code, 值: {'shares': int, 'buy_price': float, 'buy_date': str}
        self.portfolio_value = initial_capital
        self.trade_log = []

        # 交易时间限制
        self.BUY_CUTOFF_TIME = time(14, 50)  # 买入截止时间
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
        start_fetch_date = (current_date - timedelta(days=10)).strftime("%Y%m%d")
        end_fetch_date = (current_date - timedelta(days=1)).strftime("%Y%m%d")

        hist_df = self.get_daily_data(stock_code, start_fetch_date, end_fetch_date)

        if hist_df.empty or len(hist_df) < 4:
            # print(f"  - [信息] {stock_code} 的历史数据不足，无法计算买入价。")
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

    def log_trade(self, date_time, stock_code, action, shares, price):
        """将一笔交易记录到交易日志中。"""
        amount = shares * price
        log_entry = {
            "交易时间": date_time,
            "股票代码": stock_code,
            "操作": action,
            "股数": shares,
            "价格": price,
            "金额": amount
        }
        self.trade_log.append(log_entry)
        action_cn = "买入" if action == "BUY" else "卖出"
        print(f"  -> {action_cn}: {stock_code} | {shares} 股 @ {price:.2f} | 金额: {amount:.2f}")

    def run(self):
        """主回测循环。"""
        current_dt = self.start_dt
        while current_dt <= self.end_dt:
            date_str = current_dt.strftime("%Y%m%d")
            print(f"\n--- 交易日: {current_dt.strftime('%Y-%m-%d')} ---")
            print(f"交易前现金: {self.cash:.2f}")

            # 跳过周末，可以通过真实的交易日历来增强
            if current_dt.weekday() >= 5:
                print("周末，跳过。")
                current_dt += timedelta(days=1)
                continue

            # --- 1. 买入阶段 ---
            print("正在扫描新的可买入股票...")
            target_stocks = self.get_target_stocks_from_cache(date_str)

            if not target_stocks:
                print("今天没有发现目标股票。")
            else:
                potential_buys = []
                for stock_code in target_stocks:
                    if self.cash < self.position_size:
                        break  # 现金不足以支撑另一笔完整交易
                    buy_target_price = self.calculate_buy_target_price(stock_code, date_str)
                    if buy_target_price is None: continue

                    min_df = self.get_minute_data(stock_code, date_str)
                    if min_df.empty: continue

                    touched_df = min_df[(min_df['low'] <= buy_target_price) & (min_df['high'] >= buy_target_price)]
                    if not touched_df.empty:
                        first_touch_time = touched_df.index[0]
                        if first_touch_time.time() < self.BUY_CUTOFF_TIME:
                            potential_buys.append({
                                'time': first_touch_time,
                                'stock': stock_code,
                                'price': buy_target_price
                            })

                potential_buys.sort(key=lambda x: x['time'])
                print(f"在 {self.BUY_CUTOFF_TIME.strftime('%H:%M')} 前发现 {len(potential_buys)} 个潜在买入机会。")
                for buy_order in potential_buys:
                    if self.cash >= self.position_size:
                        buy_price = buy_order['price']
                        shares_to_buy = math.floor((self.position_size / buy_price) / 100) * 100
                        if shares_to_buy > 0:
                            cost = shares_to_buy * buy_price
                            self.cash -= cost  # 立刻扣除现金
                            self.positions[buy_order['stock']] = {'shares': shares_to_buy, 'buy_price': buy_price,
                                                                  'buy_date': date_str}
                            self.log_trade(buy_order['time'], buy_order['stock'], "BUY", shares_to_buy, buy_price)
                        else:
                            print(f"  - [跳过] 资金不足以购买至少100股的 {buy_order['stock']} (价格: {buy_price:.2f})。")
                    else:
                        print("今日现金不足，无法进行更多买入。")
                        break

            # --- 2. 卖出阶段 ---
            # 创建一个持仓代码的副本进行迭代，因为可能会在循环中修改字典
            positions_to_check = list(self.positions.keys())
            if not positions_to_check:
                print("没有需要卖出的持仓。")
            else:
                print("正在检查持仓的卖出信号...")

                for stock_code in positions_to_check:
                    position = self.positions[stock_code]
                    # T+1规则：当天买入的股票不能在当天卖出
                    if position['buy_date'] == date_str:
                        continue

                    daily_df = self.get_daily_data(stock_code, (current_dt - timedelta(days=10)).strftime("%Y%m%d"),
                                                   date_str)
                    if daily_df.empty or len(daily_df) < 5: continue

                    ma5 = daily_df['close'].tail(5).mean()
                    min_df = self.get_minute_data(stock_code, date_str)
                    if min_df.empty: continue

                    sell_check_dt = datetime.combine(current_dt.date(), self.SELL_CHECK_TIME)
                    if sell_check_dt in min_df.index:
                        price_at_1457 = min_df.loc[sell_check_dt]['close']
                        if price_at_1457 < ma5:
                            print(f"  - [卖出信号] {stock_code}: 14:57价格 ({price_at_1457:.2f}) 低于 MA5 ({ma5:.2f})。")
                            sell_price = price_at_1457
                            shares_to_sell = position['shares']
                            self.cash += shares_to_sell * sell_price  # 增加现金
                            self.log_trade(sell_check_dt, stock_code, "SELL", shares_to_sell, sell_price)
                            del self.positions[stock_code]  # 从持仓中移除
                        else:
                            print(
                                f"  - [继续持有] {stock_code}: 14:57价格 ({price_at_1457:.2f}) 不低于 MA5 ({ma5:.2f})。")

            # --- 3. 每日总结 ---
            # 计算每日结束时的投资组合总价值
            holdings_value = 0
            for stock_code, position in self.positions.items():
                # 获取最新收盘价来为持仓估值
                eod_df = self.get_daily_data(stock_code, date_str, date_str)
                if not eod_df.empty:
                    last_close = eod_df.iloc[-1]['close']
                    holdings_value += position['shares'] * last_close
                else:  # 如果没有数据，按买入价估值（保守策略）
                    holdings_value += position['shares'] * position['buy_price']

            self.portfolio_value = self.cash + holdings_value
            print(f"交易日 {date_str} 结束:")
            print(f"  现金: {self.cash:.2f}")
            print(f"  持仓市值: {holdings_value:.2f}")
            print(f"  投资组合总值: {self.portfolio_value:.2f}")

            # 前进到下一天
            current_dt += timedelta(days=1)

        # --- 4. 最终报告 ---
        print("\n--- 回测结束 ---")
        print(f"初始资金: {self.initial_capital:.2f}")
        print(f"最终投资组合价值: {self.portfolio_value:.2f}")
        pnl = self.portfolio_value - self.initial_capital
        pnl_percent = (pnl / self.initial_capital) * 100
        print(f"总盈亏: {pnl:.2f} ({pnl_percent:.2f}%)")

        # 保存交易日志到Excel
        if self.trade_log:
            log_df = pd.DataFrame(self.trade_log)
            log_df.to_excel("qmt_trading_log.xlsx", index=False)
            print("\n交易日志已保存到 'qmt_trading_log.xlsx'。")
        else:
            print("\n回测期间未执行任何交易。")


if __name__ == '__main__':
    # --- 配置区 ---
    # 注意：你必须提供一个包含有效本地数据的日期范围。
    # 示例日期仅用于演示。
    START_DATE = "20250701"
    END_DATE = "20250723"
    INITIAL_CAPITAL = 200000.0
    POSITION_SIZE_PER_TRADE = 20000.0

    # --- 运行回测 ---
    backtester = Backtester(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        position_size=POSITION_SIZE_PER_TRADE
    )
    backtester.run()