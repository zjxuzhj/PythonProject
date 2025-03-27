import os
import time

import akshare as ak
import pandas as pd


class StockQuery:
    # 类属性定义文件路径
    CSV_PATH = os.path.join(os.path.dirname(__file__), 'stock_code_name.csv')

    def __init__(self, auto_update=False):
        """
        初始化股票查询工具
        :param auto_update: 是否在初始化时自动更新数据
        """
        if auto_update or not os.path.exists(self.CSV_PATH):
            self._refresh_data()

        self._load_data()
        self._build_mapping()

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

    def get_simple_by_code(self, code):
        return code[2:]

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
    print(query_tool.get_name_by_code(add_stock_prefix("603881")))  # 输出代码对应名称
    print(query_tool.get_code_by_name("数据港"))  # 输出名称对应代码

    # 模糊查询示例
    print(query_tool.query("茅台", exact_match=False))  # 返回所有包含"茅台"的股票代码
