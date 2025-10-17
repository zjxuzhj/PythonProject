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
    
    def validate_etf_with_simulated_price(self, etf_code: str, target_date: str, simulated_price: float):
        """
        验证功能：使用模拟价格计算ETF动量分数
        
        Args:
            etf_code: ETF代码（例如：159509）
            target_date: 目标日期，格式'YYYYMMDD'（例如：20251015）
            simulated_price: 模拟价格（例如：15.0）
            
        Returns:
            包含计算结果的字典或None
        """
        # 参数验证
        validation_result = self._validate_parameters(etf_code, target_date, simulated_price)
        if not validation_result['valid']:
            print(f"参数验证失败: {validation_result['error']}")
            return None
        
        # 标准化ETF代码
        normalized_etf_code = self._normalize_etf_code(etf_code)
        if not normalized_etf_code:
            print(f"错误：ETF代码 {etf_code} 不在支持的ETF池中")
            print(f"支持的ETF代码: {', '.join([code.split('.')[0] for code in self.core.etf_pool])}")
            return None
        
        print(f"正在验证ETF: {normalized_etf_code} ({self.core.etf_names.get(normalized_etf_code, '未知')})")
        print(f"目标日期: {target_date}")
        print(f"模拟价格: {simulated_price}")
        print("连接xtdata获取历史数据...")
        
        try:
            # 获取历史数据（目标日期之前的数据）
            historical_data = self._get_historical_data_before_date(normalized_etf_code, target_date)
            if historical_data is None or len(historical_data) < self.core.momentum_days:
                print(f"错误：无法获取足够的历史数据（需要至少{self.core.momentum_days}天）")
                return None
            
            # 将模拟价格添加到历史数据
            complete_prices = self._append_simulated_price(historical_data, simulated_price)
            
            # 计算动量分数
            result = self.core.calculate_momentum_score(normalized_etf_code, complete_prices)
            
            if result:
                # 打印详细结果
                self._print_validation_result(result, target_date, simulated_price)
                return result
            else:
                print("计算动量分数失败")
                return None
                
        except Exception as e:
            print(f"验证过程中发生错误: {str(e)}")
            return None
    
    def _validate_parameters(self, etf_code: str, target_date: str, simulated_price: float) -> dict:
        """
        验证输入参数
        
        Args:
            etf_code: ETF代码
            target_date: 目标日期
            simulated_price: 模拟价格
            
        Returns:
            验证结果字典
        """
        # 验证ETF代码
        if not etf_code or not isinstance(etf_code, str):
            return {'valid': False, 'error': 'ETF代码不能为空且必须是字符串'}
        
        # 验证日期格式
        if not target_date or len(target_date) != 8:
            return {'valid': False, 'error': '日期格式错误，必须是8位数字（YYYYMMDD）'}
        
        try:
            datetime.strptime(target_date, '%Y%m%d')
        except ValueError:
            return {'valid': False, 'error': '日期格式不正确，请使用YYYYMMDD格式'}
        
        # 验证价格
        try:
            price = float(simulated_price)
            if price <= 0:
                return {'valid': False, 'error': '价格必须是大于0的数值'}
        except (ValueError, TypeError):
            return {'valid': False, 'error': '价格必须是有效的数值'}
        
        return {'valid': True, 'error': None}
    
    def _normalize_etf_code(self, etf_code: str) -> str:
        """
        标准化ETF代码，添加交易所后缀
        
        Args:
            etf_code: 原始ETF代码
            
        Returns:
            标准化后的ETF代码或None
        """
        # 如果已经包含交易所后缀，直接验证
        if '.' in etf_code:
            if etf_code in self.core.etf_pool:
                return etf_code
            else:
                return None
        
        # 尝试添加不同的交易所后缀
        possible_codes = [f"{etf_code}.SZ", f"{etf_code}.SH"]
        
        for code in possible_codes:
            if code in self.core.etf_pool:
                return code
        
        return None
    
    def _get_historical_data_before_date(self, etf_code: str, target_date: str):
        """
        获取指定日期之前的历史数据
        
        Args:
            etf_code: ETF代码
            target_date: 目标日期
            
        Returns:
            历史价格数组或None
        """
        try:
            # 计算需要获取的天数（考虑周末和节假日，获取更多天数）
            required_days = self.core.momentum_days + 10
            
            # 获取历史数据
            historical_data = self.core.get_historical_data_for_backtest(etf_code, required_days, target_date)
            
            if historical_data is None or len(historical_data) == 0:
                return None
            
            # 提取收盘价
            prices = historical_data['close'].values
            
            # 确保有足够的数据点
            if len(prices) < self.core.momentum_days:
                return None
            
            # 只取最近的momentum_days天数据
            return prices[-self.core.momentum_days:]
            
        except Exception as e:
            print(f"获取历史数据失败: {str(e)}")
            return None
    
    def _append_simulated_price(self, historical_prices, simulated_price: float):
        """
        将模拟价格添加到历史价格数据
        
        Args:
            historical_prices: 历史价格数组
            simulated_price: 模拟价格
            
        Returns:
            完整的价格数组
        """
        import numpy as np
        return np.append(historical_prices, simulated_price)
    
    def _print_validation_result(self, result: dict, target_date: str, simulated_price: float):
        """
        打印验证结果
        
        Args:
            result: 计算结果
            target_date: 目标日期
            simulated_price: 模拟价格
        """
        print(f"\n{'=' * 60}")
        print("ETF动量分数验证结果")
        print(f"{'=' * 60}")
        
        etf_name = self.core.etf_names.get(result['stock_code'], '未知')
        date_formatted = datetime.strptime(target_date, '%Y%m%d').strftime('%Y-%m-%d')
        
        print(f"ETF代码: {result['stock_code']} ({etf_name})")
        print(f"验证日期: {date_formatted}")
        print(f"模拟价格: {simulated_price:.2f}")
        print(f"历史数据天数: {result['data_points']}")
        print(f"-" * 60)
        print(f"动量得分: {result['score']:.4f}")
        print(f"年化收益率: {result['annualized_returns']:.2%}")
        print(f"R²值: {result['r2']:.4f}")
        print(f"-" * 60)
        
        if result['score'] > 0:
            print("✅ 该ETF在指定条件下具有正向动量")
        else:
            print("❌ 该ETF在指定条件下动量得分为0或负数")
            
        print(f"{'=' * 60}\n")


def main():
    """主函数，提供交互式查询"""
    query = ETFScoreQuery()
    
    while True:
        print("\n" + "="*50)
        print("ETF动量得分查询工具")
        print("="*50)
        print("1. 查询指定日期的ETF得分")
        print("2. 查询当前时间的ETF得分")
        print("3. 验证ETF动量分数（使用模拟价格）")
        print("4. 退出")
        print("-"*50)
        
        choice = input("请选择功能 (1-4): ").strip()
        
        if choice == '1':
            date_input = input("请输入日期 (格式: YYYYMMDD): ").strip()
            if date_input:
                query.query_etf_scores(date_input)
            else:
                print("日期不能为空")
        
        elif choice == '2':
            query.query_current_etf_scores()
        
        elif choice == '3':
            print("\n" + "-"*50)
            print("ETF动量分数验证功能")
            print("-"*50)
            print("请输入以下参数：")
            
            # 获取ETF代码
            etf_code = input("ETF代码 (例如: 159509): ").strip()
            if not etf_code:
                print("ETF代码不能为空")
                continue
            
            # 获取日期
            target_date = input("日期 (格式: YYYYMMDD, 例如: 20251015): ").strip()
            if not target_date:
                print("日期不能为空")
                continue
            
            # 获取模拟价格
            try:
                simulated_price_input = input("模拟价格 (例如: 15.0): ").strip()
                if not simulated_price_input:
                    print("价格不能为空")
                    continue
                simulated_price = float(simulated_price_input)
            except ValueError:
                print("价格必须是有效的数值")
                continue
            
            # 执行验证
            query.validate_etf_with_simulated_price(etf_code, target_date, simulated_price)
        
        elif choice == '4':
            print("退出程序")
            break
        
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()