"""
ETF动量分值查询类
支持输入指定日期，计算并显示所有ETF的动量分值排序
现在使用统一的ETF核心模块
"""

from datetime import datetime
from etf_momentum_core import ETFMomentumCore
import getAllStockCsv as tools


class ETFScoreQuery:
    """ETF动量分值查询类"""
    
    def __init__(self):
        """初始化查询类"""
        self.core = ETFMomentumCore()  # 使用统一的核心模块
        self.query_tool = tools.StockQuery()
    
    def query_etf_scores(self, target_date):
        """
        查询指定日期的ETF动量得分
        
        Args:
            target_date: 目标日期，格式'YYYYMMDD'
        """
        # 验证日期格式
        try:
            datetime.strptime(target_date, '%Y%m%d')
        except ValueError:
            print("错误：日期格式不正确，请使用YYYYMMDD格式")
            return
        
        print(f"正在计算 {target_date} 的ETF动量得分...")
        print(f"ETF池包含 {len(self.core.etf_pool)} 只ETF")
        print("连接xtdata...")
        
        # 使用核心模块获取ETF得分
        etf_scores = self.core.get_etf_scores_for_backtest(target_date)
        
        if not etf_scores:
            print("未能获取到任何有效的ETF得分数据")
            return
        
        # 统计信息
        valid_scores = [score for score in etf_scores if score['score'] > 0]
        print(f"\n计算完成！")
        print(f"总ETF数量: {len(self.core.etf_pool)}")
        print(f"成功计算得分: {len(etf_scores)}")
        print(f"有效得分(>0): {len(valid_scores)}")
        if valid_scores:
            print(f"最高得分: {max(score['score'] for score in valid_scores):.4f}")
        
        # 打印详细表格
        date_str = datetime.strptime(target_date, '%Y%m%d').strftime('%Y-%m-%d')
        self.core.print_etf_scores(etf_scores, date_str)
        
        # 显示买入推荐
        self._print_buy_recommendation(etf_scores, target_date)
    
    def query_current_etf_scores(self):
        """
        查询当前时间的ETF动量得分（结合实盘数据）
        """
        print("正在计算当前时间的ETF动量得分...")
        print(f"ETF池包含 {len(self.core.etf_pool)} 只ETF")
        print("连接xtdata...")
        
        # 使用核心模块获取当前ETF得分
        etf_scores = self.core.get_etf_scores_for_live()
        
        if not etf_scores:
            print("未能获取到任何有效的ETF得分数据")
            return
        
        # 统计信息
        valid_scores = [score for score in etf_scores if score['score'] > 0]
        print(f"\n计算完成！")
        print(f"总ETF数量: {len(self.core.etf_pool)}")
        print(f"成功计算得分: {len(etf_scores)}")
        print(f"有效得分(>0): {len(valid_scores)}")
        if valid_scores:
            print(f"最高得分: {max(score['score'] for score in valid_scores):.4f}")
        
        # 打印详细表格
        self.core.print_etf_scores(etf_scores)
        
        # 显示买入推荐
        current_date = datetime.now().strftime('%Y%m%d')
        self._print_buy_recommendation(etf_scores, current_date)
    
    def _print_buy_recommendation(self, etf_scores, date_str):
        """
        打印买入推荐
        
        Args:
            etf_scores: ETF得分列表
            date_str: 日期字符串
        """
        print(f"\n{'=' * 60}")
        print("ETF轮动策略买入推荐")
        print(f"{'=' * 60}")
        
        # 获取最佳ETF
        top_etf = self.core.get_top_etf(etf_scores)
        
        if top_etf:
            etf_name = self.core.etf_names.get(top_etf['stock_code'], '未知')
            print(f"推荐买入ETF: {top_etf['stock_code']} ({etf_name})")
            print(f"动量得分: {top_etf['score']:.4f}")
            print(f"年化收益率: {top_etf['annualized_returns']:.2%}")
            print(f"R²值: {top_etf['r2']:.4f}")
            print(f"当前价格: {top_etf['current_price']:.2f}")
            print(f"推荐理由: 在{date_str}具有最高的动量得分")
        else:
            print("当前没有合适的ETF可供买入")
            print("原因: 所有ETF的动量得分都为0或负数")
        
        print(f"{'=' * 60}\n")
    
    def get_etf_pool(self):
        """获取ETF池"""
        return self.core.etf_pool.copy()
    
    def get_etf_names(self):
        """获取ETF名称映射"""
        return self.core.etf_names.copy()


def main():
    """主函数，提供交互式查询"""
    query = ETFScoreQuery()
    
    while True:
        print("\n" + "="*50)
        print("ETF动量得分查询工具")
        print("="*50)
        print("1. 查询指定日期的ETF得分")
        print("2. 查询当前时间的ETF得分")
        print("3. 退出")
        print("-"*50)
        
        choice = input("请选择功能 (1-3): ").strip()
        
        if choice == '1':
            date_input = input("请输入日期 (格式: YYYYMMDD): ").strip()
            if date_input:
                query.query_etf_scores(date_input)
            else:
                print("日期不能为空")
        
        elif choice == '2':
            query.query_current_etf_scores()
        
        elif choice == '3':
            print("退出程序")
            break
        
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()