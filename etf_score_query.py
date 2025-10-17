"""
ETF动量分值查询类
支持输入指定日期，计算并显示所有ETF的动量分值排序
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import xtquant.xtdata as xtdata
import getAllStockCsv as tools

class ETFScoreQuery:
    """ETF动量分值查询类"""
    
    def __init__(self):
        """初始化查询类"""
        self.momentum_days = 25  # 动量计算天数
        self.query_tool = tools.StockQuery()
        
        # ETF池配置 - 与etf_momentum_rotation_qmt.py保持一致
        self.etf_pool = [
            # 境外
            "513100.SH",  # 纳指ETF
            "159509.SZ",  # 纳指科技ETF
            "513520.SH",  # 日经ETF
            "513030.SH",  # 德国ETF
            # 商品
            "518880.SH",  # 黄金ETF
            "159980.SZ",  # 有色ETF
            "159985.SZ",  # 豆粕ETF
            "501018.SH",  # 南方原油
            # 债券
            "511090.SH",  # 30年国债ETF
            # 国内
            "513130.SH",  # 恒生科技
            "513690.SH",  # 港股红利
            "510180.SH",  # 上证180
            "159915.SZ",  # 创业板ETF
            "510410.SH",  # 资源
            "515650.SH",  # 消费50
            "512290.SH",  # 生物医药
            "588120.SH",  # 科创100
            "515070.SH",  # 人工智能ETF
            "159851.SZ",  # 金融科技
            "159637.SZ",  # 新能源车
            "516160.SH",  # 新能源
            "159550.SZ",  # 互联网ETF
            "512710.SH",  # 军工ETF
            "159692.SZ",  # 证券
            "512480.SH",  # 半导体
        ]
        
        # ETF名称映射
        self.etf_names = {
            "513100.SH": "纳指ETF",
            "159509.SZ": "纳指科技ETF",
            "513520.SH": "日经ETF",
            "513030.SH": "德国ETF",
            "518880.SH": "黄金ETF",
            "159980.SZ": "有色ETF",
            "159985.SZ": "豆粕ETF",
            "501018.SH": "南方原油",
            "511090.SH": "30年国债ETF",
            "513130.SH": "恒生科技",
            "513690.SH": "港股红利",
            "510180.SH": "上证180",
            "159915.SZ": "创业板ETF",
            "510410.SH": "资源",
            "515650.SH": "消费50",
            "512290.SH": "生物医药",
            "588120.SH": "科创100",
            "515070.SH": "人工智能ETF",
            "159851.SZ": "金融科技",
            "159637.SZ": "新能源车",
            "516160.SH": "新能源",
            "159550.SZ": "互联网ETF",
            "512710.SH": "军工ETF",
            "159692.SZ": "证券",
            "512480.SH": "半导体",
        }
    
    def get_historical_data(self, stock_code, days, end_date):
        """获取指定日期前的历史数据"""
        try:
            # 计算开始日期（向前推更多天数以确保有足够的交易日）
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            start_dt = end_dt - timedelta(days=days + 30)  # 多取30天以确保有足够交易日
            start_date = start_dt.strftime('%Y%m%d')
            
            # 获取本地数据
            data = xtdata.get_local_data(
                field_list=['open', 'high', 'low', 'close', 'volume'],
                stock_list=[stock_code],
                period='1d',
                start_time=start_date,
                end_time=end_date,
            )
            
            if stock_code not in data or data[stock_code].empty:
                return None
            
            df = data[stock_code]
            
            # 转换时间索引
            df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d')
            
            # 过滤到指定日期（包含）
            df = df[df.index <= end_dt]
            
            # 取最后N天的数据
            if len(df) >= days:
                return df.tail(days)
            else:
                return df if len(df) > 0 else None
                
        except Exception as e:
            print(f"获取 {stock_code} 历史数据异常: {str(e)}")
            return None
    
    def get_price_at_date(self, stock_code, target_date):
        """获取指定日期的收盘价"""
        try:
            # 获取目标日期的数据
            data = xtdata.get_local_data(
                field_list=['close'],
                stock_list=[stock_code],
                period='1d',
                start_time=target_date,
                end_time=target_date,
            )
            
            if stock_code in data and not data[stock_code].empty:
                return float(data[stock_code]['close'].iloc[0])
            else:
                # 如果当天没有数据，尝试获取最近的交易日数据
                end_dt = datetime.strptime(target_date, '%Y%m%d')
                start_dt = end_dt - timedelta(days=7)  # 向前查找7天
                start_date = start_dt.strftime('%Y%m%d')
                
                data = xtdata.get_local_data(
                    field_list=['close'],
                    stock_list=[stock_code],
                    period='1d',
                    start_time=start_date,
                    end_time=target_date,
                )
                
                if stock_code in data and not data[stock_code].empty:
                    df = data[stock_code]
                    df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d')
                    target_dt = datetime.strptime(target_date, '%Y%m%d')
                    df = df[df.index <= target_dt]
                    if not df.empty:
                        return float(df['close'].iloc[-1])
                
                return None
                
        except Exception as e:
            print(f"获取 {stock_code} 在 {target_date} 的价格异常: {str(e)}")
            return None
    
    def calculate_momentum_score(self, stock_code, target_date):
        """计算指定日期的动量得分"""
        try:
            # 获取历史数据（不包含目标日期）
            target_dt = datetime.strptime(target_date, '%Y%m%d')
            prev_date = (target_dt - timedelta(days=1)).strftime('%Y%m%d')
            
            df = self.get_historical_data(stock_code, self.momentum_days, prev_date)
            if df is None or len(df) < self.momentum_days:
                return None
            
            # 获取目标日期的价格
            current_price = self.get_price_at_date(stock_code, target_date)
            if current_price is None or current_price <= 0:
                return None
            
            # 构建价格序列（历史价格 + 目标日期价格）
            prices = np.append(df['close'].values, current_price)
            
            # 计算对数价格
            y = np.log(prices)
            x = np.arange(len(y))
            
            # 设置权重（线性递增）
            weights = np.linspace(1, 2, len(y))
            
            # 加权线性回归
            slope, intercept = np.polyfit(x, y, 1, w=weights)
            
            # 计算年化收益率
            annualized_returns = math.exp(slope * 250) - 1
            
            # 计算R²
            y_pred = slope * x + intercept
            ss_res = np.sum(weights * (y - y_pred) ** 2)
            ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            # 计算得分
            score = annualized_returns * r2
            
            # 过滤近3日跌幅超过5%的ETF
            if len(prices) >= 4:
                recent_drops = [
                    prices[-1] / prices[-2],
                    prices[-2] / prices[-3], 
                    prices[-3] / prices[-4]
                ]
                if min(recent_drops) < 0.95:
                    score = 0
            
            return {
                'stock_code': stock_code,
                'etf_name': self.etf_names.get(stock_code, '未知'),
                'annualized_returns': annualized_returns,
                'r2': r2,
                'score': score,
                'current_price': current_price,
                'data_points': len(df)
            }
            
        except Exception as e:
            print(f"计算 {stock_code} 在 {target_date} 的动量得分异常: {str(e)}")
            return None
    
    def query_etf_scores(self, target_date):
        """查询指定日期的所有ETF分值排序"""
        print(f"\n{'='*60}")
        print(f"ETF动量分值查询 - {target_date}")
        print(f"{'='*60}")
        
        # 验证日期格式
        try:
            datetime.strptime(target_date, '%Y%m%d')
        except ValueError:
            print("错误：日期格式不正确，请使用YYYYMMDD格式（如：20250930）")
            return []
        
        etf_scores = []
        total_etfs = len(self.etf_pool)
        
        print(f"开始计算 {total_etfs} 只ETF的动量分值...")
        print("-" * 60)
        
        for idx, etf_code in enumerate(self.etf_pool, 1):
            print(f"[{idx:2d}/{total_etfs}] 计算 {etf_code} ({self.etf_names.get(etf_code, '未知')})")
            
            score_data = self.calculate_momentum_score(etf_code, target_date)
            if score_data:
                etf_scores.append(score_data)
            else:
                print(f"    ❌ 计算失败或数据不足")
        
        if not etf_scores:
            print("\n❌ 没有成功计算出任何ETF分值")
            return []
        
        # 按得分排序
        etf_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # 显示结果
        print(f"\n{'='*60}")
        print(f"ETF分值排序结果 ({target_date})")
        print(f"{'='*60}")
        print(f"{'排名':<4} {'代码':<12} {'名称':<12} {'分值':<8} {'年化收益':<10} {'R²':<8} {'价格':<8} {'数据点'}")
        print("-" * 60)
        
        for rank, data in enumerate(etf_scores, 1):
            print(f"{rank:<4} {data['stock_code']:<12} {data['etf_name']:<12} "
                  f"{data['score']:<8.4f} {data['annualized_returns']:<10.2%} "
                  f"{data['r2']:<8.4f} {data['current_price']:<8.2f} {data['data_points']}")
        
        # 统计信息
        valid_scores = [d for d in etf_scores if d['score'] > 0]
        print(f"\n统计信息:")
        print(f"  总ETF数量: {len(etf_scores)}")
        print(f"  有效分值: {len(valid_scores)} (分值 > 0)")
        print(f"  最高分值: {max(d['score'] for d in etf_scores):.4f}")
        print(f"  最低分值: {min(d['score'] for d in etf_scores):.4f}")
        
        if valid_scores:
            avg_score = sum(d['score'] for d in valid_scores) / len(valid_scores)
            print(f"  平均分值: {avg_score:.4f} (仅计算有效分值)")
        
        # 显示策略买入建议
        print(f"\n{'='*60}")
        print("🎯 ETF轮动策略买入建议")
        print(f"{'='*60}")
        
        if etf_scores and etf_scores[0]['score'] > 0:
            top_etf = etf_scores[0]
            print(f"📈 建议买入ETF: {top_etf['stock_code']} ({top_etf['etf_name']})")
            print(f"   动量评分: {top_etf['score']:.4f}")
            print(f"   年化收益率: {top_etf['annualized_returns']:.2%}")
            print(f"   R²值: {top_etf['r2']:.4f}")
            print(f"   当前价格: {top_etf['current_price']:.2f}")
            print(f"   选择理由: 该ETF在{target_date}具有最高的动量评分")
        elif etf_scores:
            print("⚠️  当前没有符合条件的ETF可供买入")
            print("   原因: 所有ETF的动量评分都 ≤ 0")
            print("   建议: 持有现金或等待更好的买入时机")
        else:
            print("❌ 无法获取ETF数据，请检查数据源")
        
        return etf_scores


def main():
    """主函数 - 直接配置日期查询"""
    query = ETFScoreQuery()

    target_date = "20251016"

    print(f"ETF动量分值查询工具 - 目标日期: {target_date}")
    print("=" * 40)

    try:
        if len(target_date) != 8 or not target_date.isdigit():
            print("错误：配置的日期格式不正确，请输入8位数字 (YYYYMMDD)")
            return

        # 执行查询
        results = query.query_etf_scores(target_date)

        if results:
            print(f"\n✅ 成功计算 {len(results)} 只ETF的分值")
        else:
            print("\n❌ 查询失败，请检查日期是否有效或数据是否存在")

    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")

if __name__ == "__main__":
    main()