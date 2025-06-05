import os

import pandas as pd

import getAllStockCsv as stockCsv
from position_manage.position import Position


class Portfolio:
    def __init__(self):
        self._tx_set = set()  # 交易唯一标识缓存
        self.positions = {}
        self.cash = 1000000
        self.transaction_history = []

    def load_transaction(self, transaction):
        query_tool = stockCsv.StockQuery()
        code = transaction.stock_code
        if code not in self.positions:
            self.positions[code] = Position(code, query_tool.get_name_by_code(code))
        if transaction.action == 'BUY':
            self.positions[code].buy_transactions.append(transaction)
        else:
            self.positions[code].sell_transactions.append(transaction)
        self.transaction_history.append(transaction)
        return True

    def add_transaction(self, transaction):
        # 生成交易指纹
        tx_id = f"{transaction.date}_{transaction.stock_code}_{transaction.action}_{transaction.price}_{transaction.shares}"
        if tx_id in self._tx_set:  # 重复检查
            print(f"⚠️ 跳过重复交易: {tx_id}")
            return False

        # 更新现金
        if transaction.action == 'BUY':
            self.cash -= transaction.price * transaction.shares
        else:
            self.cash += transaction.price * transaction.shares

        # 更新持仓
        query_tool = stockCsv.StockQuery()
        code = transaction.stock_code
        if code not in self.positions:
            self.positions[code] = Position(code, query_tool.get_name_by_code(code))

        if transaction.action == 'BUY':
            self.positions[code].buy_transactions.append(transaction)
        else:
            self.positions[code].sell_transactions.append(transaction)

        self.transaction_history.append(transaction)
        self._tx_set.add(tx_id)  # 记录新交易
        return True

    def get_stock_data(self, symbol):
        """带本地缓存的数据获取，返回最后一个收盘价"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(base_dir, "data_cache")
        file_name = f"stock_{symbol}_20240201.parquet"
        cache_path = os.path.join(cache_dir, file_name)
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path, engine='fastparquet')
                # 获取最后一个收盘价
                if not df.empty and 'close' in df.columns:
                    last_close = df['close'].iloc[-1]  # 使用iloc获取最后一个close值
                    return last_close  # 返回最后一个收盘价和缓存标记
                else:
                    print(f"缓存数据无效：{symbol}")
            except Exception as e:
                print(f"缓存读取失败：{e}（建议删除损坏文件：{cache_path}）")

        # 强制更新或缓存不存在时获取新数据
        print(f"数据获取失败：{symbol}")
        return None  # 返回None表示获取失败

    def get_position_report(self, print_format="console"):
        """生成带完美对齐的持仓报表"""
        try:
            # 设置Pandas全局对齐选项（关键改进）[1,2](@ref)
            pd.set_option('display.unicode.ambiguous_as_wide', True)
            pd.set_option('display.unicode.east_asian_width', True)
            pd.set_option('display.width', 160)  # 加宽至160字符

            report_data = []
            for code, pos in self.positions.items():
                if pos.total_shares <= 0:
                    continue

                current_price = 0
                # cost_price = round(pos.avg_cost, 2)
                # profit_pct = 0

                below_5ma = "否"
                try:
                    # 获取股票历史数据
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    cache_dir = os.path.join(base_dir, "data_cache")
                    file_name = f"stock_{code}_20240201.parquet"
                    cache_path = os.path.join(cache_dir, file_name)

                    if os.path.exists(cache_path):
                        df = pd.read_parquet(cache_path, engine='fastparquet')
                        if not df.empty and 'close' in df.columns and len(df) >= 5:
                            current_price = df['close'].iloc[-1]
                            # 计算五日均线，检查当前价格是否跌破五日线
                            df['5_day_MA'] = df['close'].rolling(window=5).mean()
                            below_5ma = "是" if current_price < df['5_day_MA'].iloc[-1] else "否"
                except Exception as e:
                    print(f"⚠️ 计算五日线失败({code}): {e}")
                    below_5ma = "错误"

                # 步骤 1
                original_avg_buy_cost = round(pos.avg_cost, 2)
                # 步骤 2: 计算所有买入交易的总成本
                total_buy_cost_for_position = sum(tx.price * tx.shares for tx in pos.buy_transactions)
                # 步骤 3: 计算所有卖出交易的总收入
                total_sell_revenue_for_position = sum(tx.price * tx.shares for tx in pos.sell_transactions)
                # 步骤 4: 获取剩余持仓股数
                remaining_shares = pos.total_shares  # 实际上循环开始时已经判断过 > 0
                # 步骤 5: 计算当前剩余持仓的总市值
                # 确保 current_price 是有效的数值，否则这里计算会不准确
                current_market_value_remaining = current_price * remaining_shares
                # 步骤 6: 计算该股票整个持仓历史的总盈亏
                overall_pnl = (total_sell_revenue_for_position + current_market_value_remaining) - total_buy_cost_for_position
                # 步骤 7: 计算每股剩余持仓“摊占”到的总盈亏
                profit_per_remaining_share = 0.0
                if remaining_shares > 0:  # 实际上这里总是 true，因为循环开始时已检查
                    profit_per_remaining_share = overall_pnl / remaining_shares

                # 步骤 8: 计算最终的盈亏百分比
                new_profit_pct = 0.0
                if remaining_shares > 0:  # 再次检查以防万一，并处理 original_avg_buy_cost
                    if original_avg_buy_cost > 0:
                        new_profit_pct = (profit_per_remaining_share / original_avg_buy_cost) * 100.0
                    elif original_avg_buy_cost == 0:  # 成本为0的特殊情况
                        if profit_per_remaining_share > 0:
                            new_profit_pct = float('inf')
                        elif profit_per_remaining_share < 0:  # 理论上成本为0不应有负盈利，除非股价为负
                            new_profit_pct = float('-inf')
                        # else profit_per_remaining_share is 0, so new_profit_pct remains 0.0

                profit_pct = new_profit_pct  # 使用新的计算结果

                report_data.append({
                    "股票代码": code,
                    "股票名称": pos.stock_name,
                    "持仓数量": pos.total_shares,
                    "平均成本": original_avg_buy_cost,
                    "当前价格": current_price,
                    "盈亏%": profit_pct,
                    "持有天数": pos.holding_period(),
                    "是否跌破五日线": below_5ma
                })

            if not report_data:
                print("ℹ️ 当前无持仓记录")
                return None

            df_report = pd.DataFrame(report_data)

            # 控制台美化输出
            if print_format == "console":
                def calc_width(series, extra=4):
                    return max(series.astype(str).apply(len)) + extra

                code_width = calc_width(df_report['股票代码'])
                name_width = calc_width(df_report['股票名称'])

                # 列格式化配置（关键对齐设置）[3,5](@ref)
                formatters = {
                    '股票代码': lambda x: f"{x:<{code_width}}",
                    '股票名称': lambda x: f"{x:<{name_width}}",
                    '持仓数量': lambda x: f"{int(x):>5,}",
                    '平均成本': lambda x: f"{x:>5.2f}",
                    '当前价格': lambda x: f"{x:>5.2f}",
                    '盈亏%': lambda x: f"{x:>5.2f}%",
                    '持有天数': lambda x: f"{int(x):>5}"
                }

                # 生成对齐表格
                formatted = df_report.to_string(
                    index=False,
                    formatters=formatters,
                    justify='left'
                )

                # 打印带边框的表格
                print("\n" + "=" * 80)
                print("📊 当前持仓报告:".center(70))
                print("=" * 80)
                print(formatted)
                print("=" * 80)

                # 打印汇总信息
                total_value = sum(row['当前价格'] * row['持仓数量'] for _, row in df_report.iterrows())
                print(f"\n总持仓价值: ¥{total_value:,.2f} | 可用现金: ¥{self.cash:,.2f}")
                return df_report

            return df_report

        except Exception as e:
            print(f"❌ 生成报表失败: {e}")
            return None
