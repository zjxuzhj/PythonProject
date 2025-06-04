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

                current_price = self.get_stock_data(code) or 0
                cost_price = round(pos.avg_cost, 2)
                profit_pct = ((current_price - cost_price) / cost_price * 100) if cost_price else 0

                report_data.append({
                    "股票代码": code,
                    "股票名称": pos.stock_name,
                    "持仓数量": pos.total_shares,
                    "平均成本": cost_price,
                    "当前价格": current_price,
                    "盈亏%": profit_pct,
                    "持有天数": pos.holding_period(),
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
