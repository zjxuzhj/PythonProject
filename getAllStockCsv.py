import csv
import os
import time

import akshare as ak
import numpy as np
import pandas as pd


class StockQuery:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(root_dir, "output", "stock_code_name.csv")
    REPORT_CSV_PATH = os.path.join(root_dir, "output", "merged_report_2024Q3.csv")
    POSITION_CSV_PATH = os.path.join(root_dir, "output", "position_report.csv")

    def load_position_csv(filepath):
        """
        从CSV文件读取持仓数据
        :param filepath: CSV文件路径
        :return: 持仓数据列表
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV文件不存在: {filepath}")

        positions = []
        with open(filepath, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # 转换数据类型
                row['shares'] = int(row['shares'])
                row['cost_price'] = float(row['cost_price'])
                row['current_price'] = float(row['current_price'])
                row['market_value'] = float(row['market_value'])
                row['profit'] = float(row['profit'])
                positions.append(row)
        return positions

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
        is_easy_a_kill = df['clean_code'].isin(self.blacklist['easy_a_kill'])

        return df[~is_bse & ~is_st & ~is_about_to_st & ~is_bad_shareholder & ~is_easy_a_kill]

    def __init__(self, auto_update=False):
        """
        初始化股票查询工具
        :param auto_update: 是否在初始化时自动更新数据
        """
        if auto_update or not os.path.exists(self.CSV_PATH):
            self._refresh_data()

        # 初始化时检查并添加theme列
        self._ensure_theme_column_exists()
        self._ensure_time_column_exists()
        self._load_data()
        self._build_mapping()
        self.roe_cache = None  # 类级缓存
        self._init_roe_data()  # 初始化时加载
        self.industry_cache = None  # 新增行业缓存
        self._init_industry_data()  # 新增初始化方法
        self.market_value_cache = None  # 新增流通市值缓存
        self._init_market_value_data()  # 新增初始化方法
        self.revenue_growth_cache = None  # 营业总收入增速缓存
        self.profit_growth_cache = None  # 净利润增速缓存
        self._init_growth_data()  # 新增初始化方法
        self.blacklist = {
            'about_to_st': {'600243', '002496', '600696', '003032', '300561', '000004', '002762', '603813', '001270',
                            '002816', '002848', '002713', '002898', '600421', '000595', '600355',
                            '600636', '000518', '603261', '600753', '000668', '605081', '002214', '603268', '002529',
                            '002305', '002253', '000820', '002693', '603389', '000929', '600130',
                            '603838', '002058', '301288', '000691', '300093', '600238', '600228', '688511', '603789',
                            '600892', '600193', '002076', '000638', '605199', '000627', '300344',
                            '002306', '002581'},  # 即将ST的股票代码
            'bad_shareholders': {'600107', '300379', '600200', '002329', '300237', '300010', '300052', '600187',
                                 '002360', '603517', '002055', '000546', '300311', '000683', '600576',
                                 '000793', '000903', '300128', '000698', '300527', '688076', '000952', '000821',
                                 '688130', '300173', '300518', '300878', '688053', '300437', '300366',
                                 '300472', '603825', '301372', '300147', '300998', '002789', '600110', '000413',
                                 '603869', '002505', '000040', '601106', '300462', '300280', '300343',
                                 '688671', '300175', '600165', '000608', '002898', '688184', '600190', '000518',
                                 '600080', '002365', '600530'},  # 股东劣迹股票代码
            'easy_a_kill': {'000605'}  # 容易A杀的股，少参与
        }

    def _ensure_theme_column_exists(self):
        """确保CSV文件包含theme列"""
        if os.path.exists(self.CSV_PATH):
            df = pd.read_csv(self.CSV_PATH)
            if 'theme' not in df.columns:
                df['theme'] = ''  # 添加空theme列
                df.to_csv(self.CSV_PATH, index=False)
        else:
            # 创建新文件时包含theme列
            pd.DataFrame(columns=['stock_code', 'stock_name', 'theme']).to_csv(
                self.CSV_PATH, index=False)

    def _ensure_time_column_exists(self):
        """确保CSV文件包含time列"""
        if os.path.exists(self.CSV_PATH):
            df = pd.read_csv(self.CSV_PATH)
            if 'time' not in df.columns:
                df['time'] = ''  # 添加空time列
                df.to_csv(self.CSV_PATH, index=False)
        else:
            # 创建新文件时包含time列
            pd.DataFrame(columns=['stock_code', 'stock_name', 'theme', 'time']).to_csv(
                self.CSV_PATH, index=False)

    def _init_roe_data(self):
        """预加载ROE数据到内存"""
        try:
            # 单次读取并预处理
            df = pd.read_csv(
                self.REPORT_CSV_PATH,
                dtype={'stock_code': str},  # 强制字符串类型
                usecols=['stock_code', '净资产收益率']  # 仅读取必要列
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
                self.REPORT_CSV_PATH,
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
                self.REPORT_CSV_PATH,
                dtype={'stock_code': str},
                usecols=['stock_code', '流通市值'],  # 明确指定需要加载的列
                converters={
                    '流通市值': lambda x: round(pd.to_numeric(x, errors='coerce') / 1e8, 2) if x != '-' else 0.0
                }
            )
            df['stock_code'] = self.get_simple_by_code(df['stock_code'])
            # 转换为字典加速查询（键为股票代码，值为行业）
            self.market_value_cache = df.set_index('stock_code')['流通市值'].to_dict()
        except Exception as e:
            print(f"流动市值初始化失败: {str(e)}")
            self.market_value_cache = {}

    def _init_growth_data(self):
        """预加载营收和净利润增速数据到内存"""
        try:
            df = pd.read_csv(
                self.REPORT_CSV_PATH,
                dtype={'stock_code': str},
                usecols=['stock_code', '营业总收入-同比增长', '净利润-同比增长']
            )
            # 统一股票代码格式
            df['stock_code'] = self.get_simple_by_code(df['stock_code'])

            # 数据清洗：转换百分比字符串为浮点数
            def convert_growth(val):
                if isinstance(val, str) and '%' in val:
                    return float(val.rstrip('%'))
                return float(val) if not pd.isna(val) else 0.0

            df['营业总收入-同比增长'] = df['营业总收入-同比增长'].apply(convert_growth)
            df['净利润-同比增长'] = df['净利润-同比增长'].apply(convert_growth)

            # 存入缓存字典
            self.revenue_growth_cache = df.set_index('stock_code')['营业总收入-同比增长'].to_dict()
            self.profit_growth_cache = df.set_index('stock_code')['净利润-同比增长'].to_dict()

        except Exception as e:
            print(f"增速数据初始化失败: {str(e)}")
            self.revenue_growth_cache = {}
            self.profit_growth_cache = {}

    def get_revenue_growth(self, symbol):
        """获取营收增速（直接内存读取）"""
        return self.revenue_growth_cache.get(symbol, 0.0)

    def get_profit_growth(self, symbol):
        """获取净利润增速（直接内存读取）"""
        return self.profit_growth_cache.get(symbol, 0.0)


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

    # 给财报数据列新增或更新市值
    def update_stock_market_value(self):
        # 步骤1：读取CSV时指定列名，避免列名冲突
        df_csv = pd.read_csv(self.REPORT_CSV_PATH)

        # 2. 获取全市场实时数据（含动态市盈率、市净率、总市值、流通市值）
        spot_data = ak.stock_zh_a_spot_em()

        # 步骤2：若原始列存在，先删除
        for col in ['市盈率-动态', '市净率', '总市值', '流通市值']:
            if col in df_csv.columns:
                df_csv.drop(col, axis=1, inplace=True)

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
        report_date = "20250331"  # 根据实际季度调整
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
            merged_df.to_csv("output/merged_report_2024Q3.csv", index=False)
        except Exception as e:
            print(f"接口调用失败: {str(e)}")

    def get_stock_roe(self, symbol):
        """字典直查法（O(1)时间复杂度）"""
        try:
            # 统一代码格式（处理带交易所前缀的情况）
            return self.roe_cache.get(symbol, 0.0)  # 无匹配返回None

        except Exception as e:
            print(f"ROE查询异常: {symbol} - {str(e)}")
            return 0.0  # 明确返回None代替0

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
        """字典直查法（O(1)时间复杂度）"""
        try:
            return self.market_value_cache.get(symbol, 0.0)  # 无匹配返回None

        except Exception as e:
            print(f"流通市值查询异常: {symbol} - {str(e)}")
            return 0.0  # 明确返回None代替0

    def _load_data(self):
        """加载本地数据"""
        try:
            self.df = pd.read_csv(self.CSV_PATH)
            if 'theme' not in self.df.columns:
                self.df['theme'] = ''
            if 'time' not in self.df.columns:
                self.df['time'] = ''
            # 去重处理（处理可能存在的重复数据）
            self.df = self.df.drop_duplicates(subset=['stock_code'], keep='last')
        except FileNotFoundError:
            raise RuntimeError("股票数据文件不存在，请先执行数据更新")

    # 该方法默认支持所有编码格式，只需保证包含正确的6位编码即可
    def get_name_by_code(self, code):
        # 步骤1：统一格式化股票代码
        normalized_code = self._normalize_stock_code(code)

        # 步骤2：用标准化代码查询
        result = self.query(normalized_code)

        # 步骤3：验证并返回结果
        if result['status'] == 'success' and result['data']['type'] == 'code':
            return result['data']['value']
        return "未找到对应名称"

    def _normalize_stock_code(self, code):
        """将不同格式的股票代码统一转换为sh/sz前缀格式"""
        # 清洗输入：移除非字母数字字符（保留点）
        cleaned_code = ''.join(c for c in code if c.isalnum() or c == '.').upper()

        # 情况1：带点格式（如603722.SH）
        if '.' in cleaned_code:
            code_part, exchange_part = cleaned_code.split('.')
            exchange_prefix = {
                'SH': 'sh', 'SZ': 'sz', 'BJ': 'bj', 'HK': 'hk'
            }.get(exchange_part[:2], exchange_part[:2].lower())
            return f"{exchange_prefix}{code_part}"

        # 情况2：纯数字格式（如603722）
        elif cleaned_code.isdigit():
            # 根据股票代码开头识别交易所[3,6](@ref)
            if cleaned_code.startswith(('6', '900')):  # 上交所股票
                return f"sh{cleaned_code}"
            elif cleaned_code.startswith(('0', '2', '3')):  # 深交所股票
                return f"sz{cleaned_code}"
            elif cleaned_code.startswith(('4', '8', '920')):  # 北交所股票
                return f"bj{cleaned_code}"

        # 情况3：混合格式（如sh603722或603722SH）
        elif len(cleaned_code) > 2:
            # 自动识别前缀位置（开头或结尾）
            prefix = cleaned_code[:2].lower() if cleaned_code[:2].isalpha() else cleaned_code[-2:].lower()
            # 提取数字部分
            digits = ''.join(c for c in cleaned_code if c.isdigit())
            return f"{prefix}{digits}"

        # 默认返回原格式（小写）
        return cleaned_code.lower()

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
        self.code_to_theme = dict(zip(  # 新增代码到题材的映射
            self.df['stock_code'].astype(str),
            self.df['theme']
        ))
        self.code_to_time = dict(zip(  # 新增代码到题材的映射
            self.df['stock_code'].astype(str),
            self.df['time']
        ))
        # 名称到代码（1对多）
        self.name_to_codes = self.df.groupby('stock_name')['stock_code'] \
            .apply(list).to_dict()

    def update_times(self, time_dict):
        """
        批量更新股票题材
        :param time_dict: 字典格式 {股票代码: 题材}
        """
        # 读取现有数据
        df = pd.read_csv(self.CSV_PATH)

        # 创建临时列用于更新
        df['update_time'] = df['stock_code'].map(time_dict)

        # 更新time列：优先使用新值，保留原值
        df['time'] = df['update_time'].combine_first(df['time'])

        # 清理临时列并保存
        df.drop(columns=['update_time'], inplace=True)
        df.to_csv(self.CSV_PATH, index=False)

        # 重新加载数据
        self._load_data()
        self._build_mapping()
        print(f"成功更新 {len(time_dict)} 只股票涨停时间信息")

    def get_time_by_code(self, code):
        """根据股票代码获取涨停时间"""
        time = self.code_to_time.get(str(code), "")
        # 显式判断是否为数值型NaN
        if isinstance(time, float) and np.isnan(time):
            return "其他"  # 返回原始NaN值
        return time if time else "其他"

    def add_time(self, code, time):
        """添加或更新单个股票的涨停时间"""
        self.update_times({code: time})

    def batch_add_times(self, time_list):
        """批量添加涨停时间（列表格式）"""
        time_dict = {item['code']: item['time'] for item in time_list}
        self.update_times(time_dict)

    def get_all_times(self):
        """获取所有股票的涨停时间映射"""
        return self.df.set_index('stock_code')['time'].to_dict()

    def find_stocks_by_time(self, keyword):
        """根据涨停时间关键词搜索股票"""
        return self.df[self.df['time'].str.contains(keyword, case=False, na=False)]

    def update_themes(self, theme_dict):
        """
        批量更新股票题材
        :param theme_dict: 字典格式 {股票代码: 题材}
        """
        # 读取现有数据
        df = pd.read_csv(self.CSV_PATH)

        # 创建临时列用于更新
        df['update_theme'] = df['stock_code'].map(theme_dict)

        # 更新theme列：优先使用新值，保留原值
        df['theme'] = df['update_theme'].combine_first(df['theme'])

        # 清理临时列并保存
        df.drop(columns=['update_theme'], inplace=True)
        df.to_csv(self.CSV_PATH, index=False)

        # 重新加载数据
        self._load_data()
        self._build_mapping()
        print(f"成功更新 {len(theme_dict)} 只股票的题材信息")

    def get_theme_by_code(self, code):
        """根据股票代码获取题材"""
        theme = self.code_to_theme.get(str(code), "")
        # 显式判断是否为数值型NaN
        if isinstance(theme, float) and np.isnan(theme):
            return "其他"  # 返回原始NaN值
        return theme if theme else "其他"

    def add_theme(self, code, theme):
        """添加或更新单个股票的题材"""
        self.update_themes({code: theme})

    def batch_add_themes(self, theme_list):
        """批量添加题材（列表格式）"""
        theme_dict = {item['code']: item['theme'] for item in theme_list}
        self.update_themes(theme_dict)

    def get_all_themes(self):
        """获取所有股票的题材映射"""
        return self.df.set_index('stock_code')['theme'].to_dict()

    def find_stocks_by_theme(self, keyword):
        """根据题材关键词搜索股票"""
        return self.df[self.df['theme'].str.contains(keyword, case=False, na=False)]

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

    def load_position_csv(self):
        if not os.path.exists(self.POSITION_CSV_PATH):
            raise FileNotFoundError(f"CSV文件不存在")

        try:
            # 读取CSV并转换数据类型
            df = pd.read_csv(
                self.POSITION_CSV_PATH,
                dtype={'stock_code': str},  # 股票代码保留为字符串[7](@ref)
                converters={
                    'shares': int,
                    'cost_price': float,
                    'current_price': float,
                    'market_value': float,
                    'profit': float
                }
            )
            return df.to_dict('records')  # 转换为字典列表[10](@ref)
        except Exception as e:
            print(f"❌ CSV读取失败: {str(e)}")
            raise


def convert_stock_code(original_code):
    """
    "603722.SH" -> "sh603722"
    """
    # 分割股票代码和交易所标识
    if '.' not in original_code:
        raise ValueError("股票代码格式错误，必须包含'.'分隔符")

    code_part, exchange_part = original_code.split('.')

    # 映射交易所标识到前缀
    exchange_map = {
        'SH': 'sh',  # 上海证券交易所
        'SZ': 'sz',  # 深圳证券交易所
        'BJ': 'bj',  # 北京证券交易所
        'HK': 'hk'  # 香港交易所
    }

    # 获取前缀（不区分大小写）
    prefix = exchange_map.get(exchange_part.upper())

    if prefix is None:
        # 如果遇到未知交易所，保留原格式
        return f"{exchange_part.lower()}{code_part}"

    return f"{prefix}{code_part}"


def convert_to_standard_format(compact_code):
    """
    "sh603722" -> "603722.SH"
    """
    # 验证输入格式
    if not compact_code or len(compact_code) < 3 or compact_code[:2].isnumeric():
        raise ValueError("股票代码格式错误，应以交易所前缀开头（如sh/sz）")

    # 交易所前缀映射（小写->大写交易所标识）
    exchange_prefix_map = {
        'sh': 'SH',  # 上海证券交易所[1,6](@ref)
        'sz': 'SZ',  # 深圳证券交易所[1,7](@ref)
        'bj': 'BJ',  # 北京证券交易所[2](@ref)
        'hk': 'HK'  # 香港交易所[6](@ref)
    }

    # 提取前缀（前2字符）和数字部分
    prefix = compact_code[:2].lower()
    number_part = compact_code[2:]

    # 验证数字部分（应为6位数字）[1,2](@ref)
    if not number_part.isdigit() or len(number_part) != 6:
        raise ValueError("股票代码数字部分应为6位数字")

    # 获取交易所标识（支持未知交易所）
    exchange = exchange_prefix_map.get(prefix, prefix.upper())

    return f"{number_part}.{exchange}"


def code_add_prefix(stock_code):
    """
    "603722" -> "sh603722"
    """
    code = str(stock_code).strip()

    # 验证基础格式
    if not code.isdigit() or len(code) != 6:
        raise ValueError("股票代码必须为6位数字")

    # 根据首字符判断市场
    first_digit = code[0]
    if first_digit in ('6', '5', '9'):  # 沪市包含主板/科创板/B股
        return f"sh{code}"
    elif first_digit in ('0', '3', '2'):  # 深市包含主板/创业板/B股
        return f"sz{code}"
    elif first_digit in ('8', '4'):  # 北证
        return f"bj{code}"
    else:
        raise ValueError(f"无法识别的股票代码开头：{first_digit}")


# 使用示例（单独执行此类时）
if __name__ == "__main__":
    # 初始化查询工具（自动检测数据文件是否存在）
    query_tool = StockQuery()
    query_tool.update_stock_market_value()
    # srt = "创新药"
    # themes_to_update = {
    #     'sz002162': srt,
    #     # 'sz002094': srt,
    #     # 'sz002537': srt,
    #     # 'sz300682': srt,
    #     # 'sz300368': srt,
    #     # 'sz300546': srt,
    #     # 'sz000890': srt,
    #     # 'sz002015': srt,
    #     # 'sz300468': srt,
    #     # 'sz002104': srt,
    # }
    # query_tool.update_themes(themes_to_update)
    # print("丰光精密题材:", query_tool.get_theme_by_code('bj430510'))
    # 精确查询示例
    # print(query_tool.get_name_by_code(add_stock_prefix("603881")))  # 输出代码对应名称
    # print(query_tool.get_code_by_name("数据港"))  # 输出名称对应代码

    # 模糊查询示例
    # print(query_tool.query("茅台", exact_match=False))  # 返回所有包含"茅台"的股票代码 n

    # query_tool.get_stock_roe('sh603881')
    # query_tool.addStockYjbbEm()

    # stock_yjbb_em_df = ak.stock_yjbb_em(date="20240930")
    # print(stock_yjbb_em_df)
