import os
import time

import akshare as ak
import pandas as pd
import plotly.express as px

class TopIndustry:
    # 类属性定义文件路径
    CSV_PATH = os.path.join(os.path.dirname(__file__), "top30_industries.csv")

    def __init__(self, auto_update=False):
        self.industry_cache = None  # 类级缓存
        self._init_industry_data()  # 初始化时加载

    def _init_industry_data(self):
        try:
            # 单次读取并预处理
            self.industry_cache = df = pd.read_csv(
                self.CSV_PATH)['名称'].tolist()

        except Exception as e:
            print(f"industry数据初始化失败: {str(e)}")
            self.roe_cache = {}

    def is_top30_industry(self, industry_name):
        """查询行业是否在Top30"""
        try:
            # 标准化输入（处理全角/半角字符）
            input_name = industry_name.strip().replace('　', ' ')
            return any(input_name == name.strip() for name in self.industry_cache)

        except FileNotFoundError:
            print("数据文件未找到，正在重新生成...")
            return industry_name in get_top_industry()

    @classmethod
    def get_instance(cls):
        """获取单例实例（可选）"""
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

def get_top_industry():
    """获取行业Top30数据并缓存到CSV"""
    cache_file = "top30_industries.csv"
    # 检查缓存文件有效性（30分钟有效期）
    if os.path.exists(cache_file):
        file_time = os.path.getmtime(cache_file)
        if (time.time() - file_time) < 1800:  # 30分钟有效期
            return pd.read_csv(cache_file)['名称'].tolist()

    # 获取10日行业数据
    df_10d = ak.stock_sector_fund_flow_rank(indicator="10日", sector_type="行业资金流")
    time.sleep(3)
    # 获取今日行业数据
    df_today = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")

    # 计算RPS_10（假设原始数据已包含10日涨跌幅字段）
    df_10d['RPS_10'] = df_10d['10日涨跌幅'].rank(pct=True) * 100

    # 数据合并与处理
    merged_df = pd.merge(
        df_today[['名称', '今日主力净流入-净额']],
        df_10d[['名称', 'RPS_10']],
        on='名称',
        how='inner'
    ).assign(
        主力强度=lambda x: x['今日主力净流入-净额'].rank(pct=True) * 100,
        综合强度=lambda x: x.RPS_10 * 0.6 + x.主力强度 * 0.4
    ).sort_values('综合强度', ascending=False).head(30)

    # 保存数据[1,6](@ref)
    merged_df[['名称', '综合强度']].to_csv(cache_file, index=False, encoding='utf_8_sig')
    return merged_df['名称'].tolist()


def visualize_top30():
    """生成交互式热力图[7](@ref)"""
    df = pd.read_csv("top30_industries.csv")
    fig = px.treemap(
        df,
        path=['名称'],
        values='综合强度',
        color='综合强度',
        color_continuous_scale='RdYlGn',
        title=f"行业强度Top30（更新于{time.strftime('%Y-%m-%d %H:%M')}）"
    )
    fig.write_html("top30_visual.html")
    os.startfile("top30_visual.html")  # 自动打开可视化结果



if __name__ == '__main__':
    # 示例调用
    # print(is_top30_industry("半导体"))  # 假设半导体在top30则返回True
    # visualize_top30()
    # topIndustry = TopIndustry()
    # topIndustry.is_top30_industry("半导体")

    get_top_industry()
