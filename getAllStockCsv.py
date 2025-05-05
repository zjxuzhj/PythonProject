import os
import time

import akshare as ak
import pandas as pd


class StockQuery:
    # 类属性定义文件路径
    CSV_PATH = os.path.join(os.path.dirname(__file__), 'stock_code_name.csv')
    REPORT_CSV_PATH = os.path.join(os.path.dirname(__file__), 'merged_report_2024Q3.csv')

    def get_simple_by_code(self, code):
        return code[2:]

    def get_all_filter_stocks(self):
        df_csv = pd.read_csv(self.REPORT_CSV_PATH)
        filtered_stocks = self.filter_stocks(df_csv)
        return filtered_stocks

    def filter_stocks(self, df):
        df['clean_code'] = df['stock_code'].str.extract(r'(\d{6})')[0]  # 提取纯数字代码
        is_bse = df['clean_code'].str.startswith(('43', '83', '87', '88', '92', '30', '68'))
        is_st = df['stock_name'].str.contains(r'ST|\*ST|退市', na=False)

        # 新增黑名单过滤
        is_about_to_st = df['clean_code'].isin(self.blacklist['about_to_st'])
        is_bad_shareholder = df['clean_code'].isin(self.blacklist['bad_shareholders'])

        return df[~is_bse & ~is_st & ~is_about_to_st & ~is_bad_shareholder]

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
        self.industry_cache = None  # 新增行业缓存
        self._init_industry_data()  # 新增初始化方法
        self.market_value_cache = None  # 新增流通市值缓存
        self._init_market_value_data()  # 新增初始化方法
        self.blacklist = {
            'about_to_st': {'600243', '002496','600696','003032','300561','000004','002762','603813','001270','002816','002848','002713','002898','600421','000595','600355',
                            '600636','000518','603261','600753','000668','605081','002214','603268','002529','002305','002253','000820','002693','603389','000929','600130',
                            '603838','002058','301288','000691','300093','600238','600228','688511','603789','600892','600193','002076','000638','605199','000627','300344',
                            '002306',},  # 即将ST的股票代码
            'bad_shareholders': {'600107','300379','600200','002329','300237','300010','300052','600187','002360','603517','002055','000546','300311','000683','600576',
                                 '000793','000903','300128','000698','300527','688076','000952','000821','688130','300173','300518','300878','688053','300437','300366',
                                 '300472','603825','301372','300147','300998','002789','600110','000413','603869','002505','000040','601106','300462','300280','300343',
                                 '688671','300175','600165','000608','002898','688184','600190','000518','600080','002365'}  # 股东劣迹股票代码
        }

    def _init_roe_data(self):
        """预加载ROE数据到内存[3,5](@ref)"""
        try:
            # 单次读取并预处理
            df = pd.read_csv(
                "merged_report_2024Q3.csv",
                dtype={'stock_code': str},  # 强制字符串类型[3](@ref)
                usecols=['stock_code', '净资产收益率']  # 仅读取必要列[5](@ref)
            )
            df['stock_code'] = self.get_simple_by_code(df['stock_code'])
            # 转换为字典加速查询
            self.roe_cache = df.set_index('stock_code')['净资产收益率'].to_dict()

        except Exception as e:
            print(f"ROE数据初始化失败: {str(e)}")
            self.roe_cache = {}

    def _init_industry_data(self):
        """加载行业数据到内存"""
        try:
            df = pd.read_csv(
                "merged_report_2024Q3.csv",
                dtype={'stock_code': str},
                usecols=['stock_code', '所处行业']  # 明确指定需要加载的列
            )
            df['stock_code'] = self.get_simple_by_code(df['stock_code'])
            # 转换为字典加速查询（键为股票代码，值为行业）
            self.industry_cache = df.set_index('stock_code')['所处行业'].to_dict()
        except Exception as e:
            print(f"行业数据初始化失败: {str(e)}")
            self.industry_cache = {}

    def _init_market_value_data(self):
        """加载流通市值数据到内存"""
        try:
            df = pd.read_csv(
                "merged_report_2024Q3.csv",
                dtype={'stock_code': str},
                usecols=['stock_code', '流通市值'],  # 明确指定需要加载的列
                converters={
                    '流通市值': lambda x: round(float(x) / 1e8, 2)  # 元→亿元并保留两位小数[1,4](@ref)
                }
            )
            df['stock_code'] = self.get_simple_by_code(df['stock_code'])
            # 转换为字典加速查询（键为股票代码，值为行业）
            self.market_value_cache = df.set_index('stock_code')['流通市值'].to_dict()
        except Exception as e:
            print(f"流动市值初始化失败: {str(e)}")
            self.market_value_cache = {}

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

    # 给财报数据列新增总市值和流通市值
    def addStockYjbbEm(self):
        df_csv = pd.read_csv(self.REPORT_CSV_PATH)

        # 2. 获取全市场实时数据（含动态市盈率、市净率、总市值、流通市值）
        spot_data = ak.stock_zh_a_spot_em()

        # 3. 预处理股票代码
        df_csv['clean_code'] = df_csv['stock_code'].str.extract(r'(\d{6})')[0]

        # 4. 合并数据（根据股票代码匹配）
        merged_df = pd.merge(
            df_csv,
            spot_data[['代码', '市盈率-动态', '市净率', '总市值', '流通市值']],
            left_on='clean_code',
            right_on='代码',
            how='left'
        )

        # 5. 填充目标字段
        merged_df['市盈率-动态'] = merged_df['市盈率-动态'].fillna('-')
        merged_df['市净率'] = merged_df['市净率'].fillna('-')
        merged_df['总市值'] = merged_df['总市值'].fillna('-')
        merged_df['流通市值'] = merged_df['流通市值'].fillna('-')

        # 6. 保存更新后的 CSV（删除临时列）
        merged_df.drop(['clean_code', '代码'], axis=1).to_csv(self.REPORT_CSV_PATH, index=False)

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
                'stock_code', 'stock_name', '股票代码',
                '每股收益', '营业总收入-营业总收入', '营业总收入-同比增长', '营业总收入-季度环比增长',
                '净利润-净利润', '净利润-同比增长', '净利润-季度环比增长',
                '净资产收益率', '每股经营现金流量', '销售毛利率', '所处行业', '最新公告日期'
            ]
            merged_df = merged_df[required_columns]

            # 保存结果
            merged_df.to_csv("merged_report_2024Q3.csv", index=False)
        except Exception as e:
            print(f"接口调用失败: {str(e)}")

    def get_stock_roe(self, symbol):
        """字典直查法（O(1)时间复杂度）[5](@ref)"""
        try:
            # 统一代码格式（处理带交易所前缀的情况）
            return self.roe_cache.get(symbol, 0.0)  # 无匹配返回None

        except Exception as e:
            print(f"ROE查询异常: {symbol} - {str(e)}")
            return 0.0  # 明确返回None代替0[1](@ref)

    def get_stock_industry(self, symbol):
        try:
            # 从缓存中查询（默认返回'未知行业'）
            return self.industry_cache.get(symbol, "未知行业")
        except KeyError:
            return "未知行业"
        except Exception as e:
            print(f"行业查询异常: {symbol} - {str(e)}")
            return "未知行业"

    def get_stock_market_value(self, symbol):
        """字典直查法（O(1)时间复杂度）[5](@ref)"""
        try:
            return self.market_value_cache.get(symbol, 0.0)  # 无匹配返回None

        except Exception as e:
            print(f"流通市值查询异常: {symbol} - {str(e)}")
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
    # print(query_tool.query("茅台", exact_match=False))  # 返回所有包含"茅台"的股票代码 n

    # query_tool.get_stock_roe('sh603881')
    query_tool.addStockYjbbEm()

    # stock_yjbb_em_df = ak.stock_yjbb_em(date="20240930")
    # print(stock_yjbb_em_df)
