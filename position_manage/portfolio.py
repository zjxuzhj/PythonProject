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

    def get_position_report(self):
        """生成持仓报表（带异常处理）"""
        try:
            report = []
            for code, pos in self.positions.items():
                # 跳过已清仓股票
                if pos.total_shares <= 0:
                    continue

                report.append({
                    "股票代码": code,
                    "股票名称": pos.stock_name,
                    "持仓数量": pos.total_shares,
                    "平均成本": round(pos.avg_cost, 2),
                    "持有天数": pos.holding_period(),
                    "当前价值": self.get_stock_data(code)
                })
            return pd.DataFrame(report)

        except AttributeError as e:
            print(f"属性错误: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"生成报表失败: {e}")
            return pd.DataFrame()
