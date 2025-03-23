from time import sleep

import akshare as ak
import pandas as pd

import util



pd.set_option('display.max_rows', None)  # 设置显示无限制行
pd.set_option('display.max_columns', None)  # 设置显示无限制列
pd.set_option('display.width', None)  # 自动检测控制台的宽度
pd.set_option('display.max_colwidth', 50)  # 设置列的最大宽度为50

date = "20250321"
df = ak.stock_zt_pool_em(date)
df['流通市值'] = round(df['流通市值'] / 100000000)
df['换手率'] = round(df['换手率'])

sleep(3)

# 昨日涨停池
yesDf = ak.stock_zt_pool_previous_em(date)

sleep(3)
# 跌停池
loserDf = ak.stock_zt_pool_dtgc_em(date)

# 今日涨停数
highestNum=len(df)
# 昨日涨停数
yesHighestNum=len(yesDf)
# 今日跌停数
loserNum=len(loserDf)
# 计算增长率
growth_rate = util.calculate_growth_rate(highestNum, yesHighestNum)
print(f"今日涨停数比昨天多{growth_rate}%，涨停{highestNum}家，跌停{loserNum}家")

# 检查是否存在有效数据
if not df.empty:
    # 筛选连板数≥2的股票
    lb_df = df[df['连板数'] >= 2]

    # 获取连板数量（总涨停家数）
    lb_count = len(lb_df)

    # 获取最高连板数（如果存在连板股）
    max_lb = int(lb_df['连板数'].max()) if lb_count > 0 else 0

    # 格式化输出
    print(f"连板{lb_count}家，最高{max_lb}板")
else:
    print("当日无涨停股票")

spath = f"./{date}涨停.xlsx"
df.to_excel(spath, engine='xlsxwriter')

selected_columns = ['代码', '名称', '最新价', '流通市值', '换手率', '连板数', '所属行业']
jj_df = df[selected_columns]

# 按照'连板数'列进行降序排序
sorted_temp_df = jj_df.sort_values(by='连板数', ascending=False)

sorted_temp_df_path = f"./{date}涨停排序.xlsx"
sorted_temp_df.to_excel(sorted_temp_df_path, engine='xlsxwriter')




# # 创建一个dataframe的副本
# temp_df = jj_df.copy()
#
# # 计算每个行业出现的频率，并储存在一个字典中
# industry_count = temp_df['所属行业'].value_counts().to_dict()
#
# # 使用.loc操作符添加一个新列，列的值是每个行业的频率
# temp_df.loc[:, 'industry_count'] = temp_df['所属行业'].map(industry_count)
#
# # 按照行业数量降序排列，如果行业数量相同，按照行业名称升序排列，然后按照连板数降序排列
# sorted_industry_df = temp_df.sort_values(by=['industry_count', '所属行业', '连板数'], ascending=[False, True, False])
#
# # 打印排序后的dataframe
# # print(sorted_industry_df)
# # 删除临时的 'industry_count' 列，以保持原始dataframe的结构
# temp_df = temp_df.drop(['industry_count'], axis=1)
#
# temp_path = f"./{date}涨停行业排序.xlsx"
# sorted_industry_df.to_excel(temp_path, engine='xlsxwriter')


