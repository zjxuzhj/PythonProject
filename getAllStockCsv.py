import os
import time

import akshare as ak
import pandas as pd


class StockQuery:
    # 类属性定义文件路径
    CSV_PATH = os.path.join(os.path.dirname(__file__), 'stock_code_name.csv')

    def get_simple_by_code(self, code):
        return code[2:]

    def __init__(self, auto_update=False):
        """
        初始化股票查询工具
        :param auto_update: 是否在初始化时自动更新数据
        """
        if auto_update or not os.path.exists(self.CSV_PATH):
            self._refresh_data()

        self._load_data()
        self._build_mapping()
        self.roe_cache = None  # 类级缓存
        self._init_roe_data()  # 初始化时加载

    def _init_roe_data(self):
        """预加载ROE数据到内存[3,5](@ref)"""
        try:
            # 单次读取并预处理
            df = pd.read_csv(
                "merged_report_2024Q3.csv",
                dtype={'股票代码': str},  # 强制字符串类型[3](@ref)
                usecols=['股票代码', '净资产收益率']  # 仅读取必要列[5](@ref)
            )
            # 清理股票代码（处理可能的.0后缀）
            df['股票代码'] = df['股票代码'].str.replace(r'\.0$', '', regex=True)
            # 转换为字典加速查询
            self.roe_cache = df.set_index('股票代码')['净资产收益率'].to_dict()

        except Exception as e:
            print(f"ROE数据初始化失败: {str(e)}")
            self.roe_cache = {}

    @classmethod
    def _refresh_data(cls):
        """获取并更新股票数据"""
        try:
            # 获取基础数据
            stock_info = ak.stock_info_a_code_name()
            time.sleep(10)  # API调用间隔

            # 获取实时数据
            spot_df = ak.stock_zh_a_spot()
            code_name_df = spot_df[['代码', '名称']].rename(columns={
                '代码': 'stock_code',
                '名称': 'stock_name'
            })

            # 确保目录存在
            os.makedirs(os.path.dirname(cls.CSV_PATH), exist_ok=True)

            # 保存数据（使用追加模式避免覆盖历史数据）
            header = not os.path.exists(cls.CSV_PATH)
            code_name_df.to_csv(cls.CSV_PATH, mode='a', index=False,
                                encoding='utf-8-sig', header=header)

        except Exception as e:
            print(f"数据更新失败: {str(e)}")
            raise

    # 更新财报数据
    def updateStockYjbbEm(self):
        df_csv = pd.read_csv(self.CSV_PATH)
        # 预处理股票代码
        df_csv['clean_code'] = df_csv['stock_code'].str.extract(r'(\d{6})')[0]
        # 获取业绩报表数据
        report_date = "20240930"  # 根据实际季度调整
        try:
            yjbb_df = ak.stock_yjbb_em(date=report_date)
            yjbb_df['股票代码'] = yjbb_df['股票代码'].astype(str).str.zfill(6)  # 统一为6位数字

            # 数据合并
            merged_df = pd.merge(
                df_csv,
                yjbb_df,
                left_on='clean_code',
                right_on='股票代码',
                how='left'
            )

            # 字段筛选
            required_columns = [
                'stock_code', 'stock_name','股票代码',
                '每股收益', '营业总收入-营业总收入','营业总收入-同比增长','营业总收入-季度环比增长',
                '净利润-净利润','净利润-同比增长','净利润-季度环比增长',
                '净资产收益率', '每股经营现金流量','销售毛利率','所处行业', '最新公告日期'
            ]
            merged_df = merged_df[required_columns]

            # 保存结果
            merged_df.to_csv("merged_report_2024Q3.csv", index=False)
        except Exception as e:
            print(f"接口调用失败: {str(e)}")

    # def getStockRoe(self,symbol):
    #     try:
    #         # 读取文件（注意编码格式）
    #         df = pd.read_csv("merged_report_2024Q3.csv", dtype={'股票代码': str})  # 中文编码兼容
    #         # 精确匹配目标股票（网页1示例）
    #         target_stock = df[df['股票代码'] == symbol].copy()
    #         # 中文字段名（可能存在的变体）
    #         roe_columns = [
    #             '净资产收益率',  # 直接字段（网页1显示4.01%）
    #         ]
    #         # 自动检测有效字段
    #         roe_value = target_stock['净资产收益率'].values[0]
    #         return roe_value
    #
    #     except Exception as e:
    #         return 0

    def getStockRoe(self, symbol):
        """字典直查法（O(1)时间复杂度）[5](@ref)"""
        try:
            # 统一代码格式（处理带交易所前缀的情况）
            return self.roe_cache.get(symbol, 0.0)  # 无匹配返回None

        except Exception as e:
            print(f"ROE查询异常: {symbol} - {str(e)}")
            return 0.0  # 明确返回None代替0[1](@ref)

    def _load_data(self):
        """加载本地数据"""
        try:
            self.df = pd.read_csv(self.CSV_PATH)
            # 去重处理（处理可能存在的重复数据）
            self.df = self.df.drop_duplicates(subset=['stock_code'], keep='last')
        except FileNotFoundError:
            raise RuntimeError("股票数据文件不存在，请先执行数据更新")

    def get_name_by_code(self, code):
        result = self.query(code)
        if result['status'] == 'success' and result['data']['type'] == 'code':
            return result['data']['value']
        return "未找到对应名称"

    def get_code_by_name(self, name):
        result = self.query(name)
        if result['status'] == 'success' and result['data']['type'] == 'name':
            return result['data']['value']
        return "未找到对应名称"

    def _build_mapping(self):
        """构建双向映射关系"""
        # 代码到名称（1对1）
        self.code_to_name = dict(zip(
            self.df['stock_code'].astype(str),
            self.df['stock_name']
        ))

        # 名称到代码（1对多）
        self.name_to_codes = self.df.groupby('stock_name')['stock_code'] \
            .apply(list).to_dict()

    def query(self, input_str, exact_match=True):
        """
        执行股票代码/名称查询
        :param input_str: 输入内容（支持代码/名称）
        :param exact_match: 是否精确匹配
        :return: 匹配结果字典
        """
        result = {'status': 'not_found', 'data': None}

        # 代码查询（精确匹配）
        if input_str in self.code_to_name:
            result['status'] = 'success'
            result['data'] = {
                'type': 'code',
                'value': self.code_to_name[input_str]
            }
            return result

        # 名称查询（精确/模糊）
        matches = []
        if exact_match:
            if input_str in self.name_to_codes:
                matches = self.name_to_codes[input_str]
        else:
            matches = [code for name, codes in self.name_to_codes.items()
                       if input_str in name for code in codes]

        if matches:
            result['status'] = 'success'
            result['data'] = {
                'type': 'name',
                'value': matches if len(matches) > 1 else matches[0]
            }

        return result

    @classmethod
    def get_instance(cls):
        """获取单例实例（可选）"""
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance


def add_stock_prefix(stock_code):
    """
    自动为股票代码添加市场前缀
    :param stock_code: 6位数字的股票代码（字符串或数字类型）
    :return: 带市场前缀的股票代码
    """
    code = str(stock_code).strip()

    # 验证基础格式
    if not code.isdigit() or len(code) != 6:
        raise ValueError("股票代码必须为6位数字")

    # 根据首字符判断市场[1](@ref)
    first_digit = code[0]
    if first_digit in ('6', '5', '9'):  # 沪市包含主板/科创板/B股
        return f"sh{code}"
    elif first_digit in ('0', '3', '2'):  # 深市包含主板/创业板/B股
        return f"sz{code}"
    else:
        raise ValueError(f"无法识别的股票代码开头：{first_digit}")


# 使用示例（单独执行此类时）
if __name__ == "__main__":
    # 初始化查询工具（自动检测数据文件是否存在）
    query_tool = StockQuery()

    # 精确查询示例
    # print(query_tool.get_name_by_code(add_stock_prefix("603881")))  # 输出代码对应名称
    # print(query_tool.get_code_by_name("数据港"))  # 输出名称对应代码

    # 模糊查询示例
    # print(query_tool.query("茅台", exact_match=False))  # 返回所有包含"茅台"的股票代码

    query_tool.getStockRoe('603881')

    # stock_yjbb_em_df = ak.stock_yjbb_em(date="20240930")
    # print(stock_yjbb_em_df)
