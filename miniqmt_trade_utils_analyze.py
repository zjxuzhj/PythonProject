from collections import defaultdict
from datetime import datetime, timedelta
import os
import pandas as pd
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from miniqmt_data_utils import get_stock_data  # 确保可以访问您的数据工具函数
import getAllStockCsv as tools
query_tool = tools.StockQuery()

def get_daily_price_data(stock_code, target_date):
    """获取指定日期的开盘价和收盘价"""
    try:
        # 转换股票代码格式
        formatted_code = tools.convert_stock_code(stock_code)

        # 获取股票数据（假设get_stock_data返回包含历史数据的DataFrame）
        df, _ = get_stock_data(formatted_code, isNeedLog=False)

        # 确保数据包含目标日期
        target_date = pd.Timestamp(target_date)
        if target_date not in df.index:
            # 尝试查找最近的数据
            nearby_dates = df.index.sort_values()
            closest_date = nearby_dates[nearby_dates <= target_date].max()
            if pd.isnull(closest_date):
                return None, None, None
            return df.loc[closest_date, 'open'], df.loc[closest_date, 'close'], closest_date.strftime('%Y-%m-%d')

        # 返回目标日期的开盘价和收盘价
        open_price = df.loc[target_date, 'open']
        close_price = df.loc[target_date, 'close']
        return open_price, close_price, target_date.strftime('%Y-%m-%d')

    except Exception as e:
        print(f"获取 {stock_code} 数据失败: {str(e)}")
        return None, None, None


def calculate_price_gaps(csv_path, target_date='2025-07-04'):
    """
    分析触发挂单数据，计算各层级价格相对于当日开盘价的涨幅
    参数:
        csv_path: 触发挂单CSV文件路径
        target_date: 要分析的日期
    """
    # 读取触发挂单数据
    df = pd.read_csv(csv_path)

    # 转换比率列为数值类型
    df['ratio'] = df['ratio'].str.replace('%', '').astype(float) / 100

    # 按股票和价格排序（价格从低到高）
    df = df.sort_values(['stock_code', 'price'])

    # 用于存储结果
    results = []

    # 按股票分组处理
    for stock_code, group in df.groupby('stock_code'):
        # 确保每组至少有三个层级
        if len(group) < 3:
            continue

        # 获取当天的开盘价和收盘价
        open_price, close_price, actual_date = get_daily_price_data(stock_code, target_date)
        if open_price is None or pd.isna(open_price):
            continue

        # 提取三个层级的价格并排序
        prices = group['price'].sort_values().values
        tier1 = prices[0]  # 最低价层级
        tier2 = prices[1]  # 中间价层级
        tier3 = prices[2]  # 最高价层级

        # 计算各层级相对于开盘价的涨幅
        tier1_rise = (tier1 - open_price) / open_price * 100
        tier2_rise = (tier2 - open_price) / open_price * 100
        tier3_rise = (tier3 - open_price) / open_price * 100

        # 计算层级间涨幅差异
        tier1_to_tier2_rise_diff = tier2_rise - tier1_rise
        tier2_to_tier3_rise_diff = tier3_rise - tier2_rise
        tier1_to_tier3_rise_diff = tier3_rise - tier1_rise

        # 检查层级间涨幅差异是否小于1%
        low_gap = False
        if abs(tier1_to_tier2_rise_diff) < 1 or abs(tier2_to_tier3_rise_diff) < 1:
            low_gap = True

        # 添加结果
        results.append({
            'code': query_tool.get_name_by_code(stock_code),
            'open': open_price,
            'close': close_price,
            '数据日期': actual_date,
            'level1_price': tier1,
            'level2_price': tier2,
            'level3_price': tier3,
            'level1(%)': tier1_rise,
            'level2(%)': tier2_rise,
            'level3(%)': tier3_rise,
            'level1→2(%)': tier1_to_tier2_rise_diff,
            'level2→3(%)': tier2_to_tier3_rise_diff,
            'level1→3(%)': tier1_to_tier3_rise_diff,
            'level_diff': low_gap
        })

    # 创建结果DataFrame
    result_df = pd.DataFrame(results)

    # 筛选层级间涨幅差小于1%的股票
    low_gap_df = result_df[result_df['level_diff'] == True]

    return result_df, low_gap_df


def visualize_results(result_df, low_gap_df, output_dir):
    """可视化分析结果"""
    plt.figure(figsize=(14, 8))

    # 绘制层级涨幅分布
    plt.subplot(2, 2, 1)
    plt.hist(result_df['level1(%)'], bins=30, alpha=0.5, label='level1')
    plt.hist(result_df['level2(%)'], bins=30, alpha=0.5, label='level2')
    plt.hist(result_df['level3(%)'], bins=30, alpha=0.5, label='level3')
    plt.title('各层级相对于开盘价的涨幅分布')
    plt.xlabel('(%)')
    plt.ylabel('num')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())

    # 绘制层级间涨幅差分布
    plt.subplot(2, 2, 2)
    plt.hist(result_df['level1→2(%)'], bins=30, alpha=0.5, label='level1→2')
    plt.hist(result_df['level2→3(%)'], bins=30, alpha=0.5, label='level2→3')
    plt.title('level_diff')
    plt.xlabel('(%)')
    plt.ylabel('num')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())

    # 标记小涨幅差股票
    plt.subplot(2, 2, 3)
    plt.scatter(result_df['level1→2(%)'],
                result_df['level2→3(%)'],
                alpha=0.6, label='正常')

    if not low_gap_df.empty:
        plt.scatter(low_gap_df['level1→2(%)'],
                    low_gap_df['level2→3(%)'],
                    color='red', label='level_diff')

    plt.title('层级间涨幅差分布')
    plt.xlabel('level1→2(%)')
    plt.ylabel('level2→3(%)')
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=-1, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=-1, color='gray', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 收盘价与层级价格比较
    plt.subplot(2, 2, 4)
    for i, row in result_df.iterrows():
        plt.plot([1, 2, 3],
                 [row['level1_price'], row['level2_price'], row['level3_price']],
                 marker='o', alpha=0.2, color='gray')

    if not low_gap_df.empty:
        for i, row in low_gap_df.iterrows():
            plt.plot([1, 2, 3],
                     [row['level1_price'], row['level2_price'], row['level3_price']],
                     marker='o', linewidth=2, label=row['code'])

    plt.title('层级价格结构 (收盘价参考线)')
    plt.axhline(y=result_df['close'].mean(), color='red', linestyle='--', label='平均收盘价')
    plt.xlabel('level')
    plt.ylabel('price')
    plt.xticks([1, 2, 3], ['level1', 'level2', 'level3'])
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'trigger_price_analysis.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return plot_path


def save_analysis_results(result_df, low_gap_df, output_dir):
    """
    保存分析结果到CSV文件
    参数:
        result_df: 完整分析结果DataFrame
        low_gap_df: 层级间小涨幅差股票DataFrame
        output_dir: 输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    full_result_path = os.path.join(output_dir, f'trigger_price_full_analysis_.csv')
    low_gap_result_path = os.path.join(output_dir, f'trigger_price_low_gap_analysis_.csv')

    # 保存结果
    result_df.to_csv(full_result_path, index=False, encoding='utf-8-sig')
    low_gap_df.to_csv(low_gap_result_path, index=False, encoding='utf_8_sig')

    return full_result_path, low_gap_result_path


def print_low_gap_summary(low_gap_df):
    """
    打印层级间小涨幅差股票摘要
    参数:
        low_gap_df: 层级间小涨幅差股票DataFrame
    """
    if low_gap_df.empty:
        print("\n⚠️ 未找到层级间涨幅差小于1%的股票")
        return

    print("\n=== 层级间涨幅差小于1%的股票 ===")
    print(f"共发现 {len(low_gap_df)} 只股票存在层级间小涨幅差情况")

    # 格式化打印表格
    summary_df = low_gap_df[['code', 'open', 'level1_price', 'level2_price', 'level3_price',
                             'level1→2(%)', 'level2→3(%)']].copy()

    # 高亮层级间小涨幅差
    def highlight_low_gap(row):
        low_gaps = []
        if abs(row['level1→2(%)']) < 1:
            low_gaps.append(f"level1→2: {row['level1→2(%)']:.2f}%")
        if abs(row['level2→3(%)']) < 1:
            low_gaps.append(f"level2→3: {row['level2→3(%)']:.2f}%")
        return ", ".join(low_gaps)

    summary_df['level_diff'] = summary_df.apply(highlight_low_gap, axis=1)

    # 打印表格
    print(summary_df.to_string(index=False, formatters={
        'open': '{:.2f}'.format,
        'level1_price': '{:.2f}'.format,
        'level2_price': '{:.2f}'.format,
        'level3_price': '{:.2f}'.format,
        'level1→2(%)': '{:.2f}%'.format,
        'level2→3(%)': '{:.2f}%'.format
    }))

    # 打印统计信息
    avg_tier1_to_tier2 = low_gap_df['level1→2(%)'].mean()
    avg_tier2_to_tier3 = low_gap_df['level2→3(%)'].mean()

    print("\n📊 层级间小涨幅差股票统计:")
    print(f"平均层级1→2涨幅差: {avg_tier1_to_tier2:.2f}%")
    print(f"平均层级2→3涨幅差: {avg_tier2_to_tier3:.2f}%")
    print(f"最小层级间涨幅差: {low_gap_df[['level1→2(%)', 'level2→3(%)']].abs().min().min():.2f}%")
    print(f"最大层级间涨幅差: {low_gap_df[['level1→2(%)', 'level2→3(%)']].abs().max().max():.2f}%")


def main():
    # 输入文件路径
    csv_path = "output/trigger_prices_2025-07-05.csv"

    # 验证文件是否存在
    if not os.path.exists(csv_path):
        print(f"⚠️ 文件不存在: {csv_path}")
        return

    # 分析触发挂单数据
    print(f"🔍 正在分析触发挂单数据: {csv_path}")
    result_df, low_gap_df = calculate_price_gaps(csv_path)

    if result_df.empty:
        print("⚠️ 未找到有效数据进行分析")
        return

    # 保存分析结果
    output_dir = "analysis_results"
    full_result_path, low_gap_result_path = save_analysis_results(result_df, low_gap_df, output_dir)

    # 可视化结果
    # plot_path = visualize_results(result_df, low_gap_df, output_dir)

    print("\n✅ 分析完成!")
    print(f"完整分析结果已保存至: {full_result_path}")
    print(f"层级间小涨幅差股票分析结果已保存至: {low_gap_result_path}")
    # print(f"分析图表已保存至: {plot_path}")

    # 打印层级间小涨幅差股票摘要
    print_low_gap_summary(low_gap_df)

    # 策略优化建议
    print("\n💡 策略优化建议:")
    print("1. 对于层级间涨幅差小于1%的股票，考虑调整风险偏好设置")
    print("2. 高风险偏好设置（RISK_LEVEL='high'）可扩大层级间涨幅差异")
    print("3. 增加层级数量（4-5层）可分散触发风险")
    print("4. 对于高价股（>100元），可适当放宽涨幅差阈值（1.5%-2%）")


if __name__ == "__main__":
    main()