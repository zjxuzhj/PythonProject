import logging
import time

import pandas as pd

import getStockDepart
import os


def setup_logger():
    logger = logging.getLogger('stock_analysis')
    if logger.handlers:  # 关键判断：避免重复配置
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 关闭层级传播

    # 控制台Handler（仅添加一次）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()


# 过滤北交所和ST股票
def filter_stocks(df):
    df['clean_code'] = df['stock_code'].str.extract(r'(\d{6})')[0]  # 提取纯数字代码
    is_bse = df['clean_code'].str.startswith(('43', '83', '87', '88', '92'))
    is_st = df['stock_name'].str.contains(r'ST|\*ST|退市', na=False)
    return df[~is_bse & ~is_st]


def save_temp(data, temp_file):
    """临时保存（无格式）"""
    pd.DataFrame(data).to_excel(temp_file, index=False, engine='openpyxl')


def save_final(data, output_file, temp_file):
    """最终保存（带格式）"""
    result_df = pd.DataFrame(data)
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        result_df.to_excel(writer, index=False, sheet_name='背离信号')
        # 应用格式设置...
    os.remove(temp_file)  # 删除临时文件


# 带频率控制的批量处理
def batch_process(stock_list, batch_size=1, delay=3, output_file='signals.xlsx'):
    start_date = '20240901'
    temp_file = output_file.replace('.xlsx', '_temp.xlsx')
    results = []

    # 加载已有进度
    try:
        existing_df = pd.read_excel(temp_file)
        results = existing_df.to_dict('records')
        logger.info(f"恢复已处理数据：{len(results)}条")
    except FileNotFoundError:
        pass

    # 获取当前日期（去除时分秒）
    current_date = pd.Timestamp.now().normalize()
    # 计算时间窗口边界
    start_date = current_date - pd.Timedelta(days=5)
    end_date = current_date

    total = len(stock_list)
    logger.info(f" 开始批量处理，共{total}只股票，批次大小{batch_size}")

    for i in range(0, 2000, batch_size):
    # for i in range(0, len(stock_list), batch_size):
    # for i in range(1349, 1370, batch_size):
        batch = stock_list[i:i + batch_size]
        for code, name in batch:
            try:
                logger.info(f"⚪ 开始处理 {code} {name}")
                # 获取数据
                df = getStockDepart.get_stock_data(code, start_date)
                # 计算MACD
                macd_df = getStockDepart.calculate_macd(df)
                signals = getStockDepart.detect_divergence(macd_df)
                logger.debug(f"🔍 检测到{len(signals)}条信号")
                # recent_signals = signals.last('3D')  # 近三天信号
                if not signals.empty:
                    # 筛选时间窗口内的信号
                    recent_signals = signals.loc[start_date:end_date]
                    logger.info(f"🚩 发现近5天信号：{len(recent_signals)}条（总信号{len(signals)}条）")
                else:
                    recent_signals = pd.DataFrame()

                if not recent_signals.empty:
                    results.append({
                        '代码': code,
                        '名称': name,
                        '最新信号日期': recent_signals.index[-1].strftime('%Y-%m-%d'),
                        '信号类型': '预顶' if recent_signals['预顶'].any() else '预底',
                    })
            except Exception as e:
                logger.error(f"处理{code}出错，已保存当前进度")
                save_temp(results, temp_file)  # 立即保存
                raise

        # 每批次保存
        save_temp(results, temp_file)
        time.sleep(delay)  # 频率控制
    # 最终保存
    save_final(results, output_file, temp_file)
    return pd.DataFrame(results)


if __name__ == '__main__':
    setup_logger()
    # 加载股票列表并过滤
    all_stocks = pd.read_csv('stock_code_name.csv')
    filtered_stocks = filter_stocks(all_stocks)

    # 分批处理
    result_df = batch_process(filtered_stocks[['stock_code', 'stock_name']].values)

    # 格式化输出
    writer = pd.ExcelWriter('signals.xlsx', engine='xlsxwriter')
    result_df.to_excel(writer, index=False, sheet_name='背离信号')

    # 设置Excel格式
    workbook = writer.book
    format_red = workbook.add_format({'font_color': '#FF0000'})
    format_green = workbook.add_format({'font_color': '#00B050'})

    worksheet = writer.sheets['背离信号']
    worksheet.conditional_format('D2:D1000', {
        'type': 'text',
        'criteria': 'containing',
        'value': '顶',
        'format': format_red
    })
    worksheet.conditional_format('D2:D1000', {
        'type': 'text',
        'criteria': 'containing',
        'value': '底',
        'format': format_green
    })
    writer.close()
