# Backtrader 回测系统使用说明

## 概述

本项目将原有的 `qmt_real_time_backtest.py` 回测逻辑完整迁移到 Backtrader 框架中，保持核心策略算法不变，提供更专业的回测功能和分析报告。

## 文件结构

```
backtrader_system/
├── backtrader_data_feed.py      # 数据适配器
├── backtrader_strategy.py       # 核心策略类
├── backtrader_analyzers.py      # 自定义分析器
├── backtrader_main.py          # 主执行脚本
├── backtrader_validation.py    # 验证测试脚本
├── backtrader_config.py        # 配置文件
└── BACKTRADER_README.md        # 使用说明
```

## 核心组件

### 1. 数据适配器 (backtrader_data_feed.py)

- **XtQuantDataFeed**: 将 xtquant 数据格式转换为 Backtrader 兼容格式
- **DataManager**: 管理单只股票的数据获取和指标计算
- **MultiStockDataManager**: 管理多只股票的数据源

主要功能：
- OHLCV 数据映射
- 技术指标计算 (MA5, MA10, MA20, MA30, MA60, MA120)
- 涨跌停价格计算
- 日期格式转换

### 2. 核心策略 (backtrader_strategy.py)

- **FirstLimitUpStrategy**: 继承自 `backtrader.Strategy` 的主策略类

核心功能：
- 选股逻辑集成
- 买卖信号处理
- 仓位管理
- 风险控制
- 交易记录

### 3. 分析器 (backtrader_analyzers.py)

提供多种专业分析器：
- **TradeAnalyzer**: 交易统计分析
- **TimeBasedTradeAnalyzer**: 按时间段分析交易表现
- **PerformanceAnalyzer**: 绩效指标计算
- **MonthlyReturnAnalyzer**: 月度收益分析
- **DetailedReportAnalyzer**: 综合报告生成

### 4. 主执行脚本 (backtrader_main.py)

- **BacktraderBacktester**: 整合所有组件的主回测器

## 快速开始

### 1. 基本使用

```python
from backtrader_main import BacktraderBacktester

# 创建回测器
backtester = BacktraderBacktester(
    initial_cash=1000000,
    commission=0.0003,
    start_date='20240101',
    end_date='20241231'
)

# 配置和运行
backtester.setup_cerebro()
stock_list = ['000001.SZ', '000002.SZ', '600000.SH']
backtester.add_data_feeds(stock_list)
results = backtester.run_backtest()

# 查看结果
backtester.print_detailed_results(results)
backtester.plot_results()
```

### 2. 使用配置文件

```python
from backtrader_config import BacktraderConfig, QuickConfigs

# 使用默认配置
config = BacktraderConfig()

# 或使用快速配置模板
config = QuickConfigs.small_capital_test()

# 应用配置
backtester = BacktraderBacktester(**config.get_data_params())
```

### 3. 运行验证测试

```python
from backtrader_validation import ValidationTester

# 创建验证器
validator = ValidationTester(test_params, tolerance=0.02)

# 运行验证
results = validator.run_validation()
validator.print_validation_report()
```

## 配置参数

### 基本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| initial_cash | 1000000 | 初始资金 |
| commission | 0.0003 | 手续费率 |
| start_date | '20240101' | 开始日期 |
| end_date | '20241231' | 结束日期 |

### 策略参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| position_size | 50000 | 每笔交易金额 |
| max_positions | 20 | 最大持仓数量 |
| sell_ma5_days | 3 | 跌破MA5卖出天数 |
| sell_loss_pct | 0.08 | 止损百分比 |

### 时间控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| buy_start_time | '09:30:00' | 买入开始时间 |
| buy_end_time | '14:50:00' | 买入结束时间 |
| sell_start_time | '09:30:00' | 卖出开始时间 |
| sell_end_time | '14:50:00' | 卖出结束时间 |

## 分析报告

### 1. 交易统计

- 总交易次数
- 胜率
- 盈亏比
- 平均盈利/亏损

### 2. 绩效指标

- 总收益率
- 年化收益率
- 最大回撤
- 夏普比率
- 波动率
- 卡尔马比率

### 3. 时间段分析

按买入时间段分析：
- 09:30-10:00
- 10:00-11:30
- 13:00-14:50

### 4. 月度收益

每月收益率统计和可视化

## 可视化功能

### 1. K线图

```python
backtester.plot_results()
```

包含：
- K线主图
- 买卖信号标记
- 技术指标叠加
- 成交量

### 2. 净值曲线

通过分析器自动生成投资组合净值曲线

## 性能优化

### 1. 数据预加载

```python
cerebro = bt.Cerebro(preload=True, runonce=True)
```

### 2. 批量数据处理

使用 `MultiStockDataManager` 批量管理多只股票数据

### 3. 内存优化

合理设置 `exactbars` 参数控制内存使用

## 验证机制

### 1. 结果一致性验证

```python
# 对比原始系统和Backtrader版本结果
validator = ValidationTester(test_params)
results = validator.run_validation()
```

### 2. 边界条件测试

- 小资金测试
- 大资金测试
- 短期测试
- 极端行情测试

### 3. 容差检查

设置合理容差范围（默认2%），确保迁移准确性

## 常见问题

### Q1: 如何添加新的技术指标？

在 `XtQuantDataFeed` 类中添加新的 `lines` 定义：

```python
class XtQuantDataFeed(bt.feeds.PandasData):
    lines = ('ma5', 'ma10', 'new_indicator',)  # 添加新指标
    
    params = (
        ('new_indicator', -1),  # 添加参数映射
    )
```

### Q2: 如何修改选股逻辑？

修改 `FirstLimitUpStrategy` 类中的 `_get_target_stocks_from_cache` 方法

### Q3: 如何添加新的分析器？

继承 `bt.Analyzer` 类创建自定义分析器：

```python
class CustomAnalyzer(bt.Analyzer):
    def __init__(self):
        # 初始化
        pass
    
    def next(self):
        # 每日调用
        pass
    
    def get_analysis(self):
        # 返回分析结果
        return {}
```

### Q4: 如何处理数据缺失？

数据适配器会自动处理缺失数据，使用前值填充或跳过

## 注意事项

1. **数据依赖**: 确保 xtquant 数据接口正常工作
2. **内存使用**: 大量股票回测时注意内存消耗
3. **时间对齐**: 确保所有股票数据时间对齐
4. **手续费设置**: 根据实际情况调整手续费率
5. **滑点考虑**: 在实盘交易中考虑滑点影响

## 扩展功能

### 1. 多策略回测

```python
cerebro.addstrategy(Strategy1)
cerebro.addstrategy(Strategy2)
```

### 2. 参数优化

```python
cerebro.optstrategy(FirstLimitUpStrategy, 
                   sell_loss_pct=[0.05, 0.08, 0.10])
```

### 3. 实时交易

集成实时数据源，支持实盘交易

## 更新日志

- v1.0.0: 完成基础迁移，支持核心策略逻辑
- v1.1.0: 添加自定义分析器和详细报告
- v1.2.0: 完善验证机制和配置管理

## 技术支持

如有问题或建议，请联系开发团队或提交 Issue。