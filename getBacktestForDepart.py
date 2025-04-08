import time

import numpy as np
import pandas as pd
from tqdm import tqdm  # 进度条支持
import os

import getAllStockCsv as stockCsv
import getStockDepart as depart
import getTopIndustry as industry
import sys
import subprocess

### 分析当前底背离筛选股票列表的收益率

STRATEGY_CONFIG = {
    'holding_periods': [1, 3, 5, 10],  # 动态收益周期配置[3,5](@ref)
    'dynamic_stop_loss': {
        'max_holding_days': 10,  # 最大持仓天数[4](@ref)
        'first_day_stop': -0.03,  # 首日止损阈值[4](@ref)
        'trailing_stop_ratio': 0.02  # 移动止损回撤比例[4](@ref)
    }
}


def calculate_returns(df, periods):
    """动态周期收益率计算[3](@ref)"""
    for n in periods:
        df[f'return_{n}d'] = df['close'].pct_change(n)
    return df


def format_percentage(df, columns, decimal=2):
    """保持数值类型，仅修改显示格式"""
    for col in columns:
        df[col] = df[col].apply(
            lambda x: round(x, decimal + 2) if pd.notnull(x) else np.nan
        )
    return df


# 新增统计函数（添加在format_percentage函数下方）
def calculate_performance(df, periods):
    """动态生成评估指标[3,5](@ref)"""
    # 生成周期列名列表
    period_cols = [f'{n}日收益' for n in periods] + ['动态止损收益']

    stats = {}

    # 新增最大回撤计算函数 [1,4,8](@ref)
    def max_drawdown(returns_series):
        """优化后的最大回撤计算方法（O(n)时间复杂度）"""
        cumulative_returns = (1 + returns_series).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    # 遍历所有收益周期
    for period in period_cols:
        returns = df[period].dropna()

        # 算术平均收益率[2,5](@ref)
        arithmetic_mean = returns.mean()

        # 几何平均收益率[2,5](@ref)
        geometric_mean = (returns + 1).prod() ** (1 / len(returns)) - 1 if len(returns) > 0 else np.nan

        # 年化收益率（假设10日周期，每年交易25次）[6,8](@ref)
        annualized_return = (1 + arithmetic_mean) ** 25 - 1 if not np.isnan(arithmetic_mean) else np.nan

        # 胜率计算[6,8](@ref)
        win_trades = (returns > 0).sum()
        total_trades = len(returns)
        win_rate = win_trades / total_trades if total_trades > 0 else 0

        # 盈亏比计算[9,11](@ref)
        avg_win = returns[returns > 0].mean()
        avg_loss = abs(returns[returns < 0].mean())
        profit_ratio = avg_win / avg_loss if avg_loss != 0 else np.nan

        # 夏普比率计算[12,13,14](@ref)
        excess_returns = returns - 0  # 假设无风险利率为0
        sharpe_ratio = excess_returns.mean() / excess_returns.std() if len(excess_returns) > 1 else np.nan

        # 新增最大回撤计算 [4,7](@ref)
        mdd = max_drawdown(returns) if len(returns) > 0 else np.nan
        print(f'最大回撤:{mdd}')
        stats[period] = {
            '算术平均': f"{arithmetic_mean:.2%}",
            '几何平均': f"{geometric_mean:.2%}" if not np.isnan(geometric_mean) else "N/A",
            '年化收益': f"{annualized_return:.2%}" if not np.isnan(annualized_return) else "N/A",
            '胜率': f"{win_rate:.2%}",
            '盈亏比': f"{profit_ratio:.2f}:1" if not np.isnan(profit_ratio) else "N/A",
            '夏普比率': f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "N/A",
            '最大回撤': f"{mdd:.2%}" if not np.isnan(mdd) else "N/A",
            '持股周期': f"{period}天" if period != '动态止损收益' \
                else f"{df['持有天数'].mean():.1f}天"
        }

        # 新增动态周期天数映射
        if period != '动态止损收益':
            days = int(period.replace('日收益', ''))
        else:
            days = df['持有天数'].mean()

        stats[period]['持股周期'] = f"{days:.1f}天"
    return pd.DataFrame(stats).T.reset_index().rename(columns={'index': '周期'})


def batch_process(input_path, output_path):
    """批量处理主函数"""
    # 读取输入文件
    df_input = pd.read_excel(input_path)

    # 从配置读取参数
    periods = STRATEGY_CONFIG['holding_periods']
    stop_config = STRATEGY_CONFIG['dynamic_stop_loss']
    # 初始化结果容器
    results = []

    # 创建股票查询工具实例
    query_tool = stockCsv.StockQuery()
    industry_tooll = industry.TopIndustry()

    # 遍历所有股票代码（带进度条）
    for _, row in tqdm(df_input.iterrows(), total=len(df_input)):
        symbol = row['代码']
        stock_name = query_tool.get_name_by_code(symbol)

        try:
            # 获取历史数据（新增未来10日数据获取）
            df, _ = depart.get_stock_data(symbol, start_date='20240201')  # 获取全量数据
            df_future = df.sort_index().copy()  # 确保时间升序排列

            # 技术指标计算
            df = depart.calculate_moving_averages(df)
            macd_df = depart.calculate_macd(df)
            df = depart.calculate_returns(macd_df, periods)  # 动态传入周期参数
            signals = depart.detect_divergence(query_tool, symbol, df, 60, True)  # 60日回溯期

            # 结果格式化（新增动态止损计算）
            for date, signal_row in signals.iterrows():
                if not pd.isna(signal_row['预底']):
                    buy_price = df.loc[date, 'close']
                    buy_date = pd.Timestamp(date)

                    # 获取未来10日数据（网页6动态止损原理）
                    future_data = df_future.loc[pd.Timestamp(date) + pd.Timedelta(days=1):].iloc[:5]

                    # 初始化止损参数（网页4移动止损策略）
                    exit_price = buy_price  # 默认持有到期
                    max_profit = 0.0  # 跟踪最高盈利
                    trigger_reason = "到期平仓"  # 记录触发原因

                    # 逐日检查止损条件
                    for i, (trade_date, row) in enumerate(future_data.iterrows()):
                        current_close = row['close']
                        current_return = (current_close / buy_price) - 1

                        # 条件1：首日基础止损（网页7的-3%止损铁律）
                        if current_return <= -0.03:
                            exit_price = current_close
                            trigger_reason = "3%固定止损"
                            break

                        # 条件2：移动止损（网页2的动态止损逻辑）
                        if current_return > max_profit:
                            max_profit = current_return

                        if max_profit >= 0.05:
                            trailing_stop = max_profit - 0.02
                            if current_return <= trailing_stop:
                                exit_price = current_close
                                trigger_reason = "移动止损"
                                break

                        # 条件3：强制平仓日（网页10的期限控制）
                        if i == 4:  # 第10个交易日（含买入日）
                            exit_price = current_close
                            break

                    # 计算动态收益率
                    dynamic_return = (exit_price / buy_price) - 1

                    record = {
                        '代码': symbol,
                        '名称': stock_name,
                        '信号日期': date.strftime('%Y-%m-%d'),
                        '买入价': df.loc[date, 'close'],
                        **{f'{n}日收益': df.loc[date, f'return_{n}d'] for n in periods},
                        '动态止损收益': dynamic_return,
                        '持有天数': i + 1 if trigger_reason != "到期平仓" else 10,  # 新增字段[2,6](@ref)
                        '触发原因': trigger_reason,
                        '所处行业': query_tool.get_stock_industry(symbol),
                        'Top行业': industry_tooll.is_top30_industry(query_tool.get_stock_industry(symbol))
                    }
                    results.append(record)

        except Exception as e:
            print(f"\n{symbol} 处理失败：{str(e)}")
            continue

    # 结果输出前增加格式化处理
    if len(results) == 0:
        print(f"\n当前底背离标志数为空")
        return ''
    result_df = pd.DataFrame(results)
    percentage_cols = [f'{n}日收益' for n in periods] + ['动态止损收益']
    result_df = format_percentage(result_df, percentage_cols)
    # 按日期降序
    result_df = result_df.sort_values(by='信号日期', ascending=False)

    # 增加样式格式化函数
    def color_negative_green(val):
        """将负收益显示绿色，正收益显示红色"""
        try:
            if isinstance(val, str):  # 处理格式化后的百分比字符串
                num = float(val.strip('%')) / 100
            else:
                num = val

            if num < 0:
                return 'color: green; font-weight: bold'  # 网页[7]颜色设置方法
            elif num > 0:
                return 'color: red; font-weight: bold'
        except:
            return ''

    def highlight_top_industry(val):
        """根据行业排名设置颜色"""
        try:
            if val == True:  # 顶级行业
                return 'background-color: red'
            elif val == False:
                return 'background-color: #90EE90'
        except:
            return ''

    # 新增性能统计
    performance_df = calculate_performance(result_df, periods)

    # 结果输出
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        styled_df = (
            result_df.style
            .format("{:.2%}", subset=percentage_cols)  # 数值转百分比格式[6,7](@ref)
            .map(color_negative_green, subset=percentage_cols)  # 应用颜色
            .map(highlight_top_industry, subset=['Top行业'])  # 新增此行[3](@ref)
        )
        styled_df.to_excel(writer, sheet_name='收益报告', index=False, engine='openpyxl')

        # 获取工作表
        worksheet = writer.sheets['收益报告']

        # 手动设置列宽（单位：字符数）
        worksheet.column_dimensions['C'].width = 12  # 信号日期列

        # 新增策略评估表[2,6](@ref)
        performance_style = (
            performance_df.style
            .format({'胜率': '{:.2%}', '盈亏比': '{:.2f}:1', '夏普比率': '{:.2f}'})
            .set_properties(**{'text-align': 'center'})
        )

        performance_style.to_excel(writer, sheet_name='策略评估', index=False)
        print(f"\n处理完成！结果已保存至：{output_path}")

        time.sleep(3)
        if sys.platform == "win32":
            # Windows环境
            os.startfile(output_path)  # 自动使用系统关联程序（若已设置WPS为默认则生效）
        else:
            # macOS/Linux环境
            subprocess.run(["open", output_path])
            # 若需指定WPS打开：
            # subprocess.run(["open", "-a", "WPS Office.app", output_path])


if __name__ == '__main__':
    batch_process(
        input_path="signals.xlsx",  # 输入文件路径
        output_path="output_20250329.xlsx"  # 带日期的输出文件名
    )
