# 克隆自聚宽文章：https://www.joinquant.com/post/62106
# 标题：1个趋势跟随策略
# 作者：chenjp

# 克隆自聚宽文章：https://www.joinquant.com/post/61141
# 标题：社区老策略是宝 越老越香
# 作者：Jellyh

# 克隆自聚宽文章：https://www.joinquant.com/post/60554
# 标题：AI优化多因子策略
# 作者：西蒙斯2025

# 沪深300指数增强策略（MACD-RSI趋势跟踪版）
# 策略核心：结合MACD趋势信号与RSI动量指标，辅以财务过滤和行业轮动
# 策略逻辑：通过多因子筛选优质股票，结合技术指标生成买卖信号，同时进行风险控制

# 导入必要的库
from kuanke.wizard import * # 聚宽量化交易框架函数库
from jqdata import * # 聚宽金融数据源
import talib                # 技术指标计算库
import numpy as np          # 数值计算库
import pandas as pd         # 数据分析库
import datetime             # 日期时间处理库


def initialize(context):
    """
    策略初始化函数，仅在回测/实盘开始时运行一次
    用于设置基准、初始参数、全局变量和调度任务
    """
    # ------------ 策略核心配置参数 ------------
    # 所有参数集中管理，便于后期调优和维护
    g.config = {
        # 基准指数与股票池配置
        # 'benchmark': '399006.XSHE',  # 业绩基准：创业板指数
        'index_code': '000300.XSHG', # 股票池来源：沪深300成分股
        'benchmark': '000300.XSHG',  # 业绩基准：沪深300指数
        # 'index_code': '000300.XSHG', # 股票池来源：沪深300成分股
        
        'min_market_cap': 400,     # 最小市值（亿元）
        'max_market_cap': 1200,    # 最大市值（亿元）
        
        # MACD技术指标参数
        'fastperiod': 9,           # 快线周期（EMA）
        'slowperiod': 21,          # 慢线周期（EMA）
        'signalperiod': 7,         # 信号线周期
        
        # RSI技术指标参数
        'rsi_period': 14,          # RSI计算周期
        'rsi_overbought': 70,      # RSI超买阈值
        'rsi_oversold': 30,        # RSI超卖阈值
        
        # 持仓与风险控制参数
        'hold_max': 10,            # 最大持仓数量
        'stop_loss': -0.08,        # 固定止损比例（-8%）
        'trail_stop_pct': 0.6,     # 跟踪止盈保留比例（60%）
        'max_single_position': 0.3, # 单只股票最大仓位比例（30%）
        'macd_threshold': 0.001,   # MACD信号强度过滤阈值
        
        # 行业轮动参数
        'industry_num': 3,         # 重点关注的强势行业数量
        'industry_lookback': 20,   # 行业强度计算回溯天数
        
        # 财务过滤参数
        'eps_filter': "eps > 0",   # 财务筛选条件（每股收益>0）
        'max_stock_pool': 200,     # 最大股票池数量（控制计算量）
        
        # 流动性过滤参数
        'min_volume': 10000000     # 最小日成交量（1000万）
    }
    
    # ------------ 初始化全局变量 ------------
    g.macd_data = {}          # 存储各股票当前MACD指标数据
    g.prev_macd_data = {}     # 存储各股票前一日MACD指标数据
    g.industry_strength = {}  # 存储各行业强度评分
    g.top_industries = []     # 存储强势行业列表
    g.max_prices = {}         # 存储持仓股票的历史最高价（用于止盈）
    g.stock_pool = []         # 每日更新的股票池
    g.prev_macd_saved_today = False  # 标记当天是否已经保存过 prev_macd_data
    g.last_save_date = None   # 记录上次保存 prev_macd_data 的日期
    
    # ------------ 系统基础设置 ------------
    set_benchmark(g.config['benchmark'])  # 设置业绩基准
    set_option('use_real_price', True)    # 使用真实价格进行回测/交易
    set_option("avoid_future_data", True)
    # 设置交易成本（模拟真实市场交易成本）
    set_order_cost(
        OrderCost(
            open_tax=0,            # 买入时不收取印花税
            close_tax=0.001,       # 卖出时收取0.1%印花税
            open_commission=0.0003, # 买入佣金0.03%
            close_commission=0.0003, # 卖出佣金0.03%
            min_commission=5       # 最低佣金5元
        ), 
        type='stock'  # 针对股票类型设置
    )
    
    # 调度每日任务（时间基于交易时间）
    run_daily(handle_morning_tasks, time='09:30') # 开盘(9:30)执行数据更新和卖出逻辑
    run_daily(check_stop_loss, time='10:30')      # 盘中(10:30)执行止损检查
    run_daily(check_stop_loss, time='14:30')      # 盘中(14:30)再次执行止损检查
    run_daily(execute_sell_at_close, time='14:55') # 【新增】尾盘(14:55)执行买入逻辑
    run_daily(execute_buy_at_close, time='14:56') # 【新增】尾盘(14:56)执行买入逻辑

# --- 开盘执行数据更新 ---
def handle_morning_tasks(context):
    """
    每日开盘后执行的逻辑
    流程：更新数据 -> 执行卖出
    """
    log.info(f"===== 开盘任务: {context.current_dt.date()} =====")
    log.info(f"账户总资产: {context.portfolio.total_value:.2f} 元")
    
    # 0. 重置每日标记，确保新的一天可以正常保存 prev_macd_data
    current_date = context.current_dt.date()
    if g.last_save_date != current_date:
        g.prev_macd_saved_today = False
        log.info(f"新的交易日开始，重置 prev_macd_saved_today 标记: {current_date}")

    update_sotck_date(context,False)
    

def update_sotck_date(context,isEnd=False):
    # 1. 更新股票池和市场数据
    update_stock_pool(context,isEnd)
    update_industry_strength(context)


def execute_sell_at_close(context):
    """
    每日尾盘14:55执行的卖出逻辑
    """
    log.info(f"===== 尾盘卖出任务: {context.current_dt.time()} =====")
    update_sotck_date(context,True)
    calculate_indicators(context)
    # 2. 执行卖出操作（先卖后买，保证资金可用）
    execute_sell(context)

# --- 用于在尾盘执行买入 ---
def execute_buy_at_close(context):
    """
    每日尾盘14:56执行的买入逻辑
    """
    log.info(f"===== 尾盘买入任务: {context.current_dt.time()} =====")
    update_sotck_date(context,True)
    calculate_indicators(context)
    execute_buy(context)
    

def update_stock_pool(context,isEnd):
    """
    更新每日股票池
    流程：基础池(沪深300成分股) -> 财务过滤 -> 流动性过滤 -> 限制数量
    """
    # 1. 获取沪深300成分股作为基础股票池
    # index_stocks = get_index_stocks(g.config['index_code'])
    all_stocks_df = get_all_securities(['stock'], date=context.current_dt)
    # 从所有A股中筛选出代码以'30'开头的创业板股票
    index_stocks = [stock for stock in all_stocks_df.index if stock.startswith('30')]
    
    if not index_stocks:  # 异常处理：如果获取失败则清空股票池
        g.stock_pool = []
        return
    
    # 1. 市值过滤：过滤出400<=市值<=1200
    filtered_by_market_cap = market_cap_filter(context, index_stocks)
    
    # 2. 财务过滤：筛选EPS>0的盈利企业（避免亏损股）
    filtered_by_finance = financial_filter(context, filtered_by_market_cap)
    
    # 3. 流动性过滤：筛选成交量达标的股票（避免流动性不足）
    filtered_by_liquidity = liquidity_filter(filtered_by_finance, context.current_dt,isEnd)
    
    # 4. 限制股票池大小（控制计算量，提高效率）
    g.stock_pool = filtered_by_liquidity[:g.config['max_stock_pool']]
    log.info(f"股票池构建完成，共 {len(g.stock_pool)} 只股票")

def market_cap_filter(context, stock_list):
    """
    市值过滤：筛选市值在指定范围内的股票
    """
    if not stock_list:
        return []
    
    try:
        # 构建市值查询
        q = query(
            valuation.code
        ).filter(
            valuation.code.in_(stock_list),
            # valuation.market_cap 的单位是 亿元
            valuation.market_cap >= g.config['min_market_cap'],
            valuation.market_cap <= g.config['max_market_cap']
        )
        
        # 执行查询
        df = get_fundamentals(q, date=context.previous_date)
        
        return df['code'].tolist()
    except Exception as e:
        log.error(f"市值过滤出错: {str(e)}")
        return stock_list # 出错时返回原始列表
        
def financial_filter(context, stock_list):
    """
    财务数据过滤：筛选符合财务条件的股票
    """
    if not stock_list:
        return []
    
    try:
        # 构建财务数据查询
        q = query(
            valuation.code,
            income.basic_eps.label('eps')
        ).filter(valuation.code.in_(stock_list))
        
        # 获取财务数据
        df = get_fundamentals(q, date=context.previous_date)
        if df.empty:
            return []
            
        # 应用财务过滤条件
        df = df.dropna(subset=['eps'])
        filter_func = lambda x: eval(g.config['eps_filter'], {'__builtins__': None}, {'eps': x})
        df = df[df['eps'].apply(filter_func)]
        
        return df['code'].tolist()
    except Exception as e:
        log.error(f"财务过滤出错: {str(e)}")
        return stock_list

def liquidity_filter(stock_list, current_dt,isEnd=False):
    """
    流动性过滤：筛选成交量达标的股票
    """
    if not stock_list:
        return []
    
    try:
        # 由于未来函数限制，end_date 都使用前一天
        end_date = current_dt - datetime.timedelta(days=1)
        start_date = end_date - datetime.timedelta(days=10)
        
        # 获取历史成交量数据
        df = get_price(
            stock_list,
            start_date=start_date,
            end_date=end_date,
            frequency='daily',
            fields=['volume'],
            panel=False
        )
        
        if df.empty or 'volume' not in df.columns:
            return []
        
        # 如果是尾盘执行（isEnd=True），需要添加当天的成交量数据
        if isEnd:
            log.info(f"尾盘执行，尝试添加当天成交量数据，当前时间: {current_dt}")
            try:
                # 获取当前成交量数据
                current_data = get_current_data()
                current_date = current_dt.date()
                
                log.info(f"获取到 {len(current_data)} 只股票的实时数据")
                log.info(f"历史数据行数（添加前）: {len(df)}")
                
                # 为每只股票添加当天的成交量数据
                current_volume_rows = []
                for stock in stock_list:
                    if stock in current_data:
                        current_volume = current_data[stock].volume
                        if current_volume > 0:  # 确保成交量有效
                            current_volume_rows.append({
                                'code': stock,
                                'time': current_date,
                                'volume': current_volume
                            })
                            log.debug(f"{stock}: 添加当天成交量 {current_volume}")
                        else:
                            log.debug(f"{stock}: 当天成交量为0或无效: {current_volume}")
                    else:
                        log.debug(f"{stock}: 未在实时数据中找到")
                
                # 将当天成交量数据拼接到历史数据中
                if current_volume_rows:
                    current_df = pd.DataFrame(current_volume_rows)
                    current_df['time'] = pd.to_datetime(current_df['time'])
                    log.info(f"当天成交量数据行数: {len(current_df)}")
                    log.info(f"当天数据示例: {current_df.head()}")
                    
                    df = pd.concat([df, current_df], ignore_index=True)
                    log.info(f"历史数据行数（添加后）: {len(df)}")
                    log.info(f"成功添加 {len(current_volume_rows)} 只股票的当天成交量数据")
                else:
                    log.warning("未能获取到有效的当天成交量数据")
                    
            except Exception as e:
                log.warning(f"获取当天成交量数据失败，使用历史数据进行过滤: {str(e)}")
        else:
            log.info(f"非尾盘执行（isEnd={isEnd}），仅使用历史数据")
        
        # 计算每只股票的平均成交量并过滤
        avg_vol = df.groupby('code')['volume'].mean()
        qualified = avg_vol[avg_vol >= g.config['min_volume']].index.tolist()
        
        return [stock for stock in stock_list if stock in qualified]
    except Exception as e:
        log.error(f"流动性过滤出错: {str(e)}")
        return stock_list

def calculate_indicators(context):
    """
    批量计算技术指标（MACD、RSI）
    """
    if not g.stock_pool:
        # 如果当天还没有保存过 prev_macd_data，则保存
        if not g.prev_macd_saved_today or g.last_save_date != context.current_dt.date():
            g.prev_macd_data = g.macd_data.copy()
            g.prev_macd_saved_today = True
            g.last_save_date = context.current_dt.date()
        g.macd_data = {}
        return
    
    # 检查当天是否已经保存过前日指标数据
    current_date = context.current_dt.date()
    if not g.prev_macd_saved_today or g.last_save_date != current_date:
        # 只在当天第一次调用时保存前日指标数据
        g.prev_macd_data = g.macd_data.copy()
        g.prev_macd_saved_today = True
        g.last_save_date = current_date
        log.info(f"首次调用 calculate_indicators，保存前日MACD数据，日期: {current_date}")
    else:
        log.info(f"当天已保存过前日MACD数据，跳过保存步骤，日期: {current_date}")
    
    # 重置当前指标数据（每次调用都需要重新计算）
    g.macd_data = {}
    
    
    try:
        end_date = context.previous_date  # 使用前一天作为结束日期
        start_date = end_date - datetime.timedelta(days=90)
        
        # 批量获取历史价格数据
        price_df = get_price(
            g.stock_pool,
            start_date=start_date,
            end_date=end_date,
            frequency='daily',
            fields=['close'],
            panel=False
        )
        
        if price_df.empty:
            return
        
        # 如果是尾盘执行，获取当前价格并拼接到历史数据中
        current_prices = {}
        try:
            # 获取当前价格数据
            current_prices = get_current_data()
            log.info("获取当前价格数据，准备拼接到历史数据中")
        except Exception as e:
            log.warning(f"获取当前价格失败，无法拼接当天数据: {str(e)}")
            current_prices = {}
            
            
        # 按股票代码分组计算指标
        for stock in g.stock_pool:
            try:
                stock_data = price_df[price_df['code'] == stock]
                if stock_data.empty:
                    continue
                    
                close_prices = stock_data['close'].values.copy()  # 复制数组以避免修改原数据
                
                # 如果是尾盘执行且能获取到当前价格，拼接当前价格到历史数据末尾
                if stock in current_prices:
                    current_price = current_prices[stock].last_price
                    if current_price > 0:  # 确保当前价格有效
                        # 将当前价格拼接到历史价格数组末尾
                        close_prices = np.append(close_prices, current_price)
                        log.debug(f"{stock}: 拼接当前价格 {current_price:.2f} 到历史数据末尾")
                
                if len(close_prices) < g.config['slowperiod'] + g.config['signalperiod']:
                    continue
                
                # 计算MACD
                macd, signal, hist = talib.MACD(
                    close_prices,
                    fastperiod=g.config['fastperiod'],
                    slowperiod=g.config['slowperiod'],
                    signalperiod=g.config['signalperiod']
                )
                
                # 计算RSI
                rsi = talib.RSI(close_prices, timeperiod=g.config['rsi_period'])
                
                # 存储指标数据
                g.macd_data[stock] = {
                    'dif': macd[-1],
                    'dea': signal[-1],
                    'macd': hist[-1],
                    'prev_macd': hist[-2] if len(hist) > 1 else 0,
                    'prev_dif': macd[-2] if len(macd) > 1 else 0,
                    'prev_dea': signal[-2] if len(signal) > 1 else 0,
                    'rsi': rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else 50,
                    'current_price': close_prices[-1]  # 使用修正后的价格
                }
            except Exception as e:
                log.warning(f"计算 {stock} 指标出错: {str(e)}")
                continue
                
    except Exception as e:
        log.error(f"批量计算指标出错: {str(e)}")
    
    # 输出MACD和RSI计算值
    if g.macd_data:
        print_macd_rsi_values(context, max_stocks=5)  # 显示前5只股票的详细指标


def print_macd_rsi_values(context, max_stocks=10):
    """
    输出MACD和RSI的计算值
    """
    if not g.macd_data:
        log.info("暂无MACD/RSI数据")
        return
    
    log.info("=" * 80)
    log.info("📊 MACD和RSI技术指标计算值")
    log.info("=" * 80)
    
    # 按股票代码排序，只显示前max_stocks只股票
    sorted_stocks = sorted(list(g.macd_data.keys()))[:max_stocks]
    
    for i, stock in enumerate(sorted_stocks, 1):
        data = g.macd_data[stock]
        
        # 获取股票名称
        try:
            stock_name = get_security_info(stock).display_name
        except:
            stock_name = "未知"
        
        log.info(f"\n📈 [{i:2d}] {stock} ({stock_name})")
        log.info("-" * 60)
        
        # MACD指标输出
        log.info("🔵 MACD指标:")
        log.info(f"   DIF (快线):     {data.get('dif', 0):.6f}")
        log.info(f"   DEA (慢线):     {data.get('dea', 0):.6f}")
        log.info(f"   MACD (柱状):    {data.get('macd', 0):.6f}")
        log.info(f"   前日DIF:        {data.get('prev_dif', 0):.6f}")
        log.info(f"   前日DEA:        {data.get('prev_dea', 0):.6f}")
        log.info(f"   前日MACD:       {data.get('prev_macd', 0):.6f}")
        
        # 判断MACD信号
        current_dif = data.get('dif', 0)
        current_dea = data.get('dea', 0)
        prev_dif = data.get('prev_dif', 0)
        prev_dea = data.get('prev_dea', 0)
        
        if current_dif > current_dea and prev_dif <= prev_dea:
            signal = "🟢 金叉信号"
        elif current_dif < current_dea and prev_dif >= prev_dea:
            signal = "🔴 死叉信号"
        elif current_dif > current_dea:
            signal = "🟡 多头排列"
        else:
            signal = "🟠 空头排列"
        
        log.info(f"   信号状态:       {signal}")
        
        # RSI指标输出
        rsi_value = data.get('rsi', 50)
        log.info(f"\n🔴 RSI指标:")
        log.info(f"   RSI值:          {rsi_value:.2f}")
        
        # 判断RSI状态
        if rsi_value >= 70:
            rsi_status = "🔴 超买区域 (≥70)"
        elif rsi_value <= 30:
            rsi_status = "🟢 超卖区域 (≤30)"
        elif rsi_value >= 50:
            rsi_status = "🟡 强势区域 (50-70)"
        else:
            rsi_status = "🟠 弱势区域 (30-50)"
        
        log.info(f"   RSI状态:        {rsi_status}")
        
        # 当前价格信息
        current_price = data.get('current_price', 0)
        if current_price > 0:
            log.info(f"\n💰 价格信息:")
            log.info(f"   当前价格:       {current_price:.2f} 元")
    
    log.info("\n" + "=" * 80)
    log.info(f"📋 共显示 {len(sorted_stocks)} 只股票的技术指标 (总计 {len(g.macd_data)} 只)")
    log.info("=" * 80)


def update_industry_strength(context):
    """
    更新行业强度评分，筛选强势行业
    """
    g.industry_strength = {}
    g.top_industries = []
    
    try:
        # 1. 获取申万一级行业列表
        industries = get_industries('sw_l1')
        if industries.empty:
            return
            
        all_industry_codes = industries.index.tolist()
        
        # --- (新增) 2. 过滤掉无效的行业代码，避免"找不到标的"错误 ---
        valid_industry_codes = []
        for code in all_industry_codes:
            # 使用 get_security_info 确认该代码是否为当前有效的标的
            if get_security_info(code) is not None:
                valid_industry_codes.append(code)
            # else:
                # log.warning(f"行业代码 {code} 无效或已弃用，已跳过。")
        
        if not valid_industry_codes:
            return

        # 3. 批量获取有效行业指数的价格数据
        end_date = context.current_dt
        start_date = end_date - datetime.timedelta(days=g.config['industry_lookback'])
        price_df = get_price(
            valid_industry_codes, # 使用过滤后的有效代码列表
            start_date=start_date,
            end_date=end_date,
            frequency='daily',
            fields=['close'],
            panel=False
        )
        
        if price_df.empty or 'close' not in price_df.columns:
            return
            
        # 4. 计算各行业强度得分
        for code in valid_industry_codes: # 遍历有效代码列表
            try:
                industry_data = price_df[price_df['code'] == code]
                if industry_data.empty:
                    continue
                    
                close_prices = industry_data['close'].values
                if len(close_prices) < 10:
                    continue
                    
                # 计算行业强度得分
                returns = np.diff(close_prices) / close_prices[:-1]
                avg_return = np.mean(returns)
                total_return = (close_prices[-1] / close_prices[0]) - 1
                
                # 综合评分
                score = (avg_return * 0.4 + total_return * 0.6) * 100
                g.industry_strength[code] = score
            except Exception as e:
                continue
        
        # 5. 筛选前N强行业
        g.top_industries = sorted(
            g.industry_strength.items(),
            key=lambda x: x[1],
            reverse=True
        )[:g.config['industry_num']]
        
        if g.top_industries:
            log.info(f"强势行业: {', '.join([f'{get_industry_name(k)}({v:.2f})' for k, v in g.top_industries])}")
            
    except Exception as e:
        log.error(f"更新行业强度出错: {str(e)}")

def execute_sell(context):
    """
    执行卖出操作
    """
    hold_stocks = list(context.portfolio.positions.keys())
    if not hold_stocks:
        return
        
    for stock in hold_stocks:
        try:
            if should_sell(context, stock):
                # 使用聚宽内置的 order_target_value 函数清仓
                order_target_value(stock, 0)
                log.info(f"[卖出] {stock} - 触发卖出条件")
                if stock in g.max_prices:
                    del g.max_prices[stock]
        except Exception as e:
            log.error(f"处理 {stock} 卖出出错: {str(e)}")

def should_sell(context, stock):
    """
    判断是否满足卖出条件
    """
    position = context.portfolio.positions.get(stock)
    if not position or position.total_amount <= 0:
        return False
        
    # 1. 计算当前收益率
    profit_pct = position.price / position.avg_cost - 1
    
    # 2. 固定止损条件
    if profit_pct <= g.config['stop_loss']:
        log.info(f"[{stock}] 触发固定止损: 收益率 {profit_pct:.2%}")
        return True
        
    # 3. 跟踪止盈条件
    current_price = position.price
    if stock not in g.max_prices or current_price > g.max_prices[stock]:
        g.max_prices[stock] = current_price
        
    max_price = g.max_prices[stock]
    max_profit = max_price / position.avg_cost - 1
    
    if max_profit > 0.08: # 只有当最大盈利超过8%时才启动跟踪止盈
        drawdown = (max_price - current_price) / max_price
        # 回撤超过最大盈利的一定比例时止盈
        if drawdown > (1 - g.config['trail_stop_pct']):
            log.info(f"[{stock}] 触发跟踪止盈: 最大盈利 {max_profit:.2%}, 当前盈利 {profit_pct:.2%}")
            return True
            
    # 4. MACD技术卖出信号
    macd_info = g.macd_data.get(stock)
    if not macd_info: # 如果获取不到指标（可能刚上市），则不根据技术指标卖出
        return False
        
    current_macd = macd_info['macd']
    prev_macd = macd_info.get('prev_macd', 0)
    
    # 卖出信号：MACD死叉
    macd_sell = (macd_info['dif'] < macd_info['dea'] and macd_info['prev_dif'] >= macd_info['prev_dea'])
    
    if macd_sell:
        log.info(f"[{stock}] 触发MACD死叉卖出信号")
        return True

    return False

def execute_buy(context):
    """
    执行买入操作
    """
    hold_stocks = context.portfolio.positions
    buy_list = get_buy_list(context)
    
    if not buy_list or len(hold_stocks) >= g.config['hold_max']:
        return
        
    # 计算可买入的股票数量和每只股票的基础金额
    num_to_buy = min(g.config['hold_max'] - len(hold_stocks), len(buy_list))
    available_cash = context.portfolio.available_cash
    
    if num_to_buy <= 0:
        return
        
    base_cash_per_stock = available_cash / num_to_buy
    
    # 获取大盘趋势因子
    market_factor = get_market_trend_factor(context)
    
    # 遍历买入列表
    for stock in buy_list[:num_to_buy]:
        if stock in hold_stocks:
            continue
            
        try:
            # 计算动态资金分配因子
            signal_strength = get_signal_strength(stock)
            industry_factor = get_industry_factor(stock)
            total_factor = market_factor * signal_strength * industry_factor
            
            # 计算最终买入金额
            max_allocate = context.portfolio.total_value * g.config['max_single_position']
            allocate_cash = base_cash_per_stock * total_factor
            final_cash = min(allocate_cash, max_allocate, available_cash) # 确保不超过可用资金
            
            if final_cash < 2000: # 忽略过小金额
                continue
                
            # 使用聚宽内置的 order_target_value 函数买入
            order_target_value(stock, final_cash)
            available_cash -= final_cash # 更新可用资金
            log.info(
                f"[买入] {stock} - 信号强度:{signal_strength:.2f}, "
                f"行业因子:{industry_factor:.2f}, 投入:{final_cash:.2f}元"
            )
        except Exception as e:
            log.error(f"处理 {stock} 买入出错: {str(e)}")

def get_buy_list(context):
    """
    生成买入候选列表
    """
    if not g.stock_pool:
        return []
        
    buy_candidates = []
    for stock in g.stock_pool:
        macd_info = g.macd_data.get(stock)
        prev_macd = g.prev_macd_data.get(stock)
        
        if not macd_info or not prev_macd:
            continue
            
        # 买入条件1：MACD金叉
        golden_cross = (
            macd_info['dif'] > macd_info['dea'] and
            prev_macd.get('dif', 999) <= prev_macd.get('dea', 999)
        )
        
        # 买入条件2：MACD柱在零轴上方且强度达标
        strong_signal = (
            macd_info['macd'] > 0 and
            macd_info['macd'] > g.config['macd_threshold']
        )
        
        # 买入条件3：RSI未超买
        rsi_ok = macd_info.get('rsi', 50) < g.config['rsi_overbought']
        
        if golden_cross and strong_signal and rsi_ok:
            score = macd_info['macd'] * get_industry_factor(stock)
            buy_candidates.append((stock, score))
    
    # 按综合评分降序排序
    buy_candidates.sort(key=lambda x: x[1], reverse=True)
    return [stock for stock, _ in buy_candidates]

def get_signal_strength(stock):
    """
    计算技术信号强度因子
    """
    macd_info = g.macd_data.get(stock)
    if not macd_info:
        return 1.0
        
    dif = macd_info['dif']
    dea = macd_info['dea']
    macd = macd_info['macd']
    
    # 标准化处理
    normalized_macd = min(abs(macd) / 0.05, 3.0)
    signal_diff = min(abs(dif - dea) / 0.02, 2.0)
    
    # 综合信号强度
    return 1.0 + (normalized_macd * 0.6 + signal_diff * 0.4) * 0.3

def get_industry_factor(stock):
    """
    获取行业强度因子
    """
    try:
        industry_info = get_industry(stock)
        if not industry_info or 'sw_l1' not in industry_info:
            return 1.0
            
        industry_code = industry_info['sw_l1']['industry_code']
        # 检查是否属于强势行业
        top_industry_codes = [code for code, _ in g.top_industries]
        if industry_code in top_industry_codes:
            return 1.3  # 强势行业股票获得30%加成
        return 1.0
    except:
        return 1.0

def get_market_trend_factor(context):
    """
    获取大盘趋势因子
    """
    try:
        # 获取沪深300指数的价格数据
        prices = get_price(
            g.config['benchmark'],
            count=g.config['slowperiod'] * 2,
            fields=['close'],
            end_date=context.current_dt,
            panel=False
        )
        
        if prices.empty or len(prices) < g.config['slowperiod']:
            return 1.0
            
        close_prices = prices['close'].values
        # 计算大盘MACD
        macd, signal, hist = talib.MACD(
            close_prices,
            fastperiod=g.config['fastperiod'],
            slowperiod=g.config['slowperiod'],
            signalperiod=g.config['signalperiod']
        )
        
        # 计算大盘RSI
        rsi = talib.RSI(close_prices, timeperiod=g.config['rsi_period'])[-1]
        
        # 根据RSI状态调整因子
        rsi_factor = 0.85 if rsi > g.config['rsi_overbought'] else \
                     1.1 if rsi < g.config['rsi_oversold'] else 1.0
        
        # 根据大盘MACD状态确定基础趋势因子
        if hist[-1] > 0 and macd[-1] > signal[-1]:
            return 1.0 * rsi_factor  # 大盘多头趋势
        elif hist[-1] < 0 and macd[-1] < signal[-1]:
            return 0.7 * rsi_factor  # 大盘空头趋势，减仓
        else:
            return 0.85 * rsi_factor # 大盘震荡趋势，略微减仓
    except:
        return 1.0

def check_stop_loss(context):
    """
    盘中止损检查 (已修正版本)
    在交易时段再次检查止损止盈条件，及时应对市场波动
    """
    # 获取所有持仓股票列表
    positions_list = list(context.portfolio.positions.keys())
    if not positions_list:
        return

    # 一次性获取所有持仓股的当前快照数据，效率更高
    current_data = get_current_data()

    for stock in positions_list:
        try:
            position = context.portfolio.positions.get(stock)
            if not position or position.total_amount <= 0:
                continue

            stock_data = current_data[stock]
            # 使用 .last_price 获取最新价，并检查股票是否停牌或价格无效
            if stock_data and not stock_data.paused and stock_data.last_price > 0:
                current_price = stock_data.last_price
            else:
                # 如果获取不到最新价（例如停牌、数据错误），则使用持仓成本价作为参考，避免出错
                current_price = position.price
                log.warning(f"无法获取 {stock} 的最新价格，使用持仓价格进行判断。")

            # 1. 固定止损检查
            profit_pct = current_price / position.avg_cost - 1
            if profit_pct <= g.config['stop_loss']:
                order_target_value(stock, 0)
                log.info(f"[盘中止损] {stock} 触发卖出，当前价格: {current_price}, 收益率: {profit_pct:.2%}")
                if stock in g.max_prices:
                    del g.max_prices[stock]
                continue  # 股票已卖出，跳过后续的止盈检查

            # 2. 跟踪止盈检查
            if stock in g.max_prices:
                # 更新当日最高价
                g.max_prices[stock] = max(g.max_prices[stock], current_price)
                max_price = g.max_prices[stock]
                max_profit = max_price / position.avg_cost - 1

                # 对收益超过12%的股票实施跟踪止盈
                if max_profit > 0.12:
                    drawdown = (max_price - current_price) / max_price
                    if drawdown > (1 - g.config['trail_stop_pct']):
                        order_target_value(stock, 0)
                        log.info(f"[盘中止盈] {stock} 触发卖出，最大盈利 {max_profit:.2%}, 当前盈利 {profit_pct:.2%}")
                        del g.max_prices[stock]
        except Exception as e:
            log.error(f"盘中检查 {stock} 出错: {str(e)}")

# (辅助函数) 获取行业中文名，用于日志输出
def get_industry_name(industry_code):
    try:
        return get_industries('sw_l1').loc[industry_code, 'name']
    except:
        return industry_code