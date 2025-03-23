import akshare as ak
import pandas as pd
from datetime import datetime

#### 获得当天排名前20的股票数据

# 获取当日全量股票数据（含成交额）
def get_stock_data(target_date):
    # 获取全市场实时行情数据[1](@ref)
    spot_df = ak.stock_zh_a_spot()

    # 获取当日日期（模拟2025-03-21数据）
    spot_df['日期'] = pd.to_datetime(target_date)

    # 筛选所需字段[1,3](@ref)
    columns_mapping = {
        '代码': '股票代码',
        '名称': '股票名称',
        '最新价': '最新价格',
        '成交额': '成交金额',
        '涨跌幅': '涨跌幅(%)'
    }
    return spot_df.rename(columns=columns_mapping)[
        ['日期', '股票代码', '股票名称', '最新价格', '成交金额', '涨跌幅(%)']]


# 主程序
if __name__ == "__main__":
    target_date = "2025-03-21"  # 指定目标日期

    # 获取数据
    stock_news_em_df = ak.stock_news_em(symbol="603777")
    print(stock_news_em_df)