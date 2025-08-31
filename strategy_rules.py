from enum import Enum


class RuleEnum(Enum):
    """
    策略规则的枚举类型。
    每个成员都包含了一个元组 (description, rule_type, message)，
    使得每个规则都自带完整的元数据。
    """

    def __new__(cls, *args, **kwds):
        # 自动为枚举成员分配一个唯一的整数值
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, description: str, rule_type: str, message: str):
        """
        初始化每个枚举成员的附加属性。
        :param description: 规则的简短中文解释。
        :param rule_type: 规则的分类（如：基本面, 形态, 量价等）。
        :param message: 具体的排除信息。
        """
        self.description = description
        self.rule_type = rule_type
        self.message = message

    # =================================================================
    # ==                 is_valid_first_limit_up_day 规则             ==
    # =================================================================

    # --- 基础检查 ---
    NO_DATA_AFTER_LIMIT_UP = ("涨停日后无数据", "基础检查", "涨停日是数据流的最后一天，无法判断后续走势")
    INSUFFICIENT_DATA_BEFORE_LIMIT_UP = ("涨停前数据不足", "基础检查", "涨停发生时，历史数据少于20天")

    # --- 基本面筛选 ---
    MARKET_CAP_TOO_HIGH = ("市值过高", "基本面", "总市值超过预设阈值（如250亿）")

    # --- 板块题材筛选 ---
    IS_FINANCE_OR_SECURITY = ("金融证券板块", "板块题材", "股票属于金融或证券板块或者外贸题材，波动逻辑不同")

    # --- 形态筛选 ---
    IS_CONSECUTIVE_LIMIT_UP = ("连板股", "形态", "涨停前一日或后一日也是涨停，非首板")
    LARGE_GAIN_BEFORE_LIMIT_UP = ("首板前涨幅过高", "形态", "首板前的5天内累计涨幅过大（>15%）")
    PRE_LIMIT_UP_CHAOTIC_PATTERN = ("涨停前K线形态混乱","形态","涨停前两日连续出现宽幅震荡且实体极小的K线，且近期未有效测试MA10支撑")
    FREQUENT_LIMIT_UPS_RECENTLY = (
        "近期涨停过于频繁",
        "形态",
        "涨停前的10个交易日内，涨停次数过多（>=4次），短期风险过高"
    )
    EXTREME_VOLATILITY_UP_DOWN = (
        "先连板后连跌",
        "形态",
        "涨停前的10日内，出现至少2连板后又出现至少2连跌停的极端走势"
    )
    LOW_VOLUME_REBOUND_AFTER_STREAK = (
        "前期连板后缩量反抽",
        "形态",
        "涨停前10日内存在连板高点，当前涨停疑似对前期高点进行缩量、弱势反抽"
    )
    ONE_WORD_LIMIT_UP_WITH_WEAK_TREND = (
        "弱势趋势下的一字板",
        "形态",
        "一字板涨停，但涨停前20日内多次跌破MA20且T-2日收盘在MA5之下，趋势不稳固"
    )
    RECENT_LIMIT_UP_FAILURE = (
        "近期涨停破位",
        "形态",
        "涨停前的4日内存在另一个涨停板，且该涨停板在后续被深度回调（跌破约一半位置）"
    )
    REJECTED_BY_RESISTANCE_PLATFORM = (
        "受阻于前期密集平台",
        "形态",
        "涨停次日（T+1）冲高回落，受阻于由前期多个高点形成的密集平台压力区"
    )

    FALSE_BREAKOUT_OF_KEY_PEAK = (
        "对关键高点的假突破",
        "形态",
        "涨停次日(T+1)缩量尝试突破前40日高点，但冲高回落形成长上影线，构成假突破形态"
    )

    ISLAND_REVERSAL_PATTERN = (
        "岛形反转形态",
        "形态",
        "涨停日跳空高开，但次日又跳空低开形成岛形反转，且涨停前无明显洗盘形态"
    )


    # --- 量价关系筛选 ---
    P1_HUGE_VOLUME_GREEN_CANDLE = ("次日巨量阴线", "量价", "涨停次日放出巨量且收阴线，抛压沉重")

    REJECTED_BY_HIGH_VOLUME_PEAK = (
        "受阻于前期放量高点",
        "量价",
        "涨停次日（T+1）缩量尝试突破前期放量高点失败，构成压力"
    )

    WEAK_FOLLOW_THROUGH_VOLUME = (
        "次日温和放量但缺乏技术支撑",
        "量价",
        "涨停次日成交量温和放大，但缺乏均线粘合、均线支撑或洗盘等强力技术形态支持"
    )

    OBV_BEARISH_DIVERGENCE = (
        "OBV顶背离",
        "量价",
        "涨停次日(T+1)股价创近期新高但OBV指标未新高，形成顶背离且无其他强支撑"
    )

    ABNORMAL_VOLUME_SPIKE = (
        "成交量异常放大",
        "量价",
        "涨停日或次日成交量，是过去120日内最大成交量的4倍以上，可能为出货天量"
    )

    # --- 市场活性筛选 ---
    INSUFFICIENT_PRIOR_VOLATILITY = (
        "前期波动率过低",
        "市场活性",
        "涨停前40日内交投不活跃（多数交易日振幅小于3.5%），且次日未出现强力放量"
    )

    # =================================================================
    # ==                 is_valid_buy_opportunity 规则                ==
    # =================================================================

    # --- 买入前检查 ---
    PRICE_TOO_HIGH = ("股价过高", "价格", "买入前一日的收盘价高于80元")
    NEW_LIMIT_UP_AFTER_FIRST = ("首板后又涨停", "形态", "在计划买入前，股票再次涨停，形态改变")
    PRICE_FELL_BELOW_LIMIT_UP_PRICE = ("跌破首板价", "形态", "调整期间股价已跌破首板日的收盘价，走弱")

    T1_GAP_DOWN_WEAK_REBOUND = (
        "次日低开弱反弹",
        "形态",
        "涨停次日大幅低开并带上影线，显示承接无力，且前期缺乏强均线支撑或粘合形态"
    )

    STALLED_BELOW_PRIOR_HIGH = (
        "涨停前受阻于前高",
        "形态",
        "涨停前3日内，股价已接近但未能突破前20日高点，显示前高压制明显"
    )

    LIMIT_UP_HITS_MAJOR_RESISTANCE = (
        "涨停撞线90日高点",
        "形态",
        "涨停价精准触及前期90日高点压力位，且无均线粘合形态配合，突破概率较低"
    )

    STRONG_T1_IMMEDIATE_FAILURE_T2 = (
        "放量大阳次日即走弱",
        "形态",
        "涨停次日(T+1)为放量大阳线，但T+2日却低开且未能收复T+1实体一半，属转弱信号"
    )

    REJECTION_AT_KEY_RESISTANCE = (
        "关键阻力位受阻回落",
        "形态",
        "T+1至T+3日反弹高点极度接近前期高点但未突破，且T+3日出现长上影线，表明压力巨大"
    )

    DOUBLE_TOP_DISTRIBUTION = (
        "T+1,T+2双顶放量滞涨",
        "形态",
        "T+1与T+2连续创40日新高但形成双顶，且两日均冲高回落、成交量巨大，为顶部出货信号"
    )

    REJECTED_BY_DOUBLE_TOP_NECKLINE = (
        "受阻于前期双顶颈线",
        "形态",
        "涨停后的整理期间，股价反弹受阻于前期双顶形态的颈线位压力"
    )

    PERSISTENT_REJECTION_LOWERING_PEAKS = (
        "连续长上影且高点回落",
        "形态",
        "T+1至T+3日连续出现长上影线，且高点回落（H1>H3>H2），表明抛压持续且多头力竭"
    )

    EXCESSIVE_GAP_UP_IN_CONSOLIDATION = (
        "整理期连续大幅高开",
        "形态",
        "T+2或T+3出现连续大幅跳空高开(>2%)或单日过高开盘(>3%)，表明情绪过热，有加速见顶风险"
    )

    PERSISTENT_WEAKNESS_IN_CONSOLIDATION = (
        "整理期持续阴跌",
        "形态",
        "T+1日冲高小幅回落，随后T+2与T+3日连续收阴，表明买方动能衰竭，空方渐强"
    )

    STRONG_GAP_UP_QUICK_FAILURE = (
        "跳空高开后继无力",
        "形态",
        "T+1日跳空高开表现强势，但T+2与T+3日连续跌破T+1最低点，属强势不再的信号"
    )

    FAILED_REBOUND_PATTERN = (
        "阴跌后反弹无力",
        "形态",
        "T+1收阴后，T+2弱势反弹，但T+3旋即跌破T+2阳线实体一半，属上涨乏力形态"
    )

    FLAT_TOP_REJECTION = (
        "三天平顶压力",
        "形态",
        "T+1至T+3日高点形成平顶，且前两日均带长上影线，表明上方压力巨大，且T+3日未表现出强势突破"
    )

    GAP_UP_STRENGTH_INVALIDATED = (
        "强势高开后支撑失效",
        "形态",
        "T+1日曾现跳空高开的极强形态，但T+2与T+3日连续跌破其最低点，强势被证伪"
    )

    # --- 趋势筛选 ---
    TREND_CONFLICT = (
        "中短期均线趋势冲突",
        "趋势",
        "短期趋势向上（MA10↑），但中期趋势向下（MA60↓），形成趋势冲突，且缺乏其他强有力的技术形态支撑"
    )

    FAILED_MA55_BREAKOUT = (
        "突破55日线失败",
        "趋势",
        "T+1日尝试突破MA55，但T+2与T+3日均收于其下为假突破，且缺乏短中期均线支撑"
    )

    REJECTED_BY_DOWNTREND_LINE = (
        "涨停撞线下降趋势线",
        "趋势",
        "涨停价精准触及由前期高点形成的下降趋势线，且缺乏其他强有力的技术形态支撑"
    )

    REJECTED_BY_MAJOR_MA = (
        "受重要均线压制",
        "趋势",
        "T+1日尝试突破关键均线但收盘受阻，且T+2日继续走弱确认压制有效"
    )

    UNSUPPORTED_PULLBACK_ON_T3 = (
        "T+3日出现无支撑的回调",
        "趋势",
        "T+3日出现中等幅度的回调（落入模拟买点区间），但该回调缺乏MA20或MA30的均线支撑，视为弱势信号"
    )

    # --- 策略逻辑规则 ---
    PRIOR_BUY_OPPORTUNITY_TRIGGERED = (
        "T+2已触发买点",
        "策略逻辑",
        "T+4买入机会被跳过，因为T+2时已满足模拟挂单买入条件，避免重复建仓"
    )