import sys
import os
import time

# 添加上级目录到系统路径，以便导入 getAllStockCsv
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from stock_manager import StockPoolManager
    from getAllStockCsv import StockQuery
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

def batch_import():
    print("正在初始化股票查询工具...")
    try:
        query_tool = StockQuery()
    except Exception as e:
        print(f"StockQuery 初始化失败，无法自动填充行业: {e}")
        query_tool = None

    manager = StockPoolManager()

    # 从截图提取的股票列表
    stocks_to_import = [
        # 第一批
        ("002819", "东方中科"),
        ("002202", "金风科技"),
        ("600509", "天富能源"),
        ("002440", "闰土股份"),
        ("600410", "华胜天成"),
        ("000801", "四川九洲"),
        ("301091", "深城交"),
        ("002915", "中欣氟材"),
        ("002407", "多氟多"),
        ("002322", "理工能科"),
        ("002135", "东南网架"),
        ("600666", "奥瑞德"),
        # 第二批
        ("002828", "贝肯能源"),
        ("000676", "智度股份"),
        ("000678", "襄阳轴承"),
        ("600845", "宝信软件"),
        ("002361", "神剑股份"),
        ("002465", "海格通信"),
        ("601179", "中国西电"),
        ("002165", "红宝丽"),
        ("601933", "永辉超市"),
        ("600879", "航天电子"),
        ("600610", "中毅达"),
        ("603308", "应流股份")
    ]

    print(f"\n准备导入 {len(stocks_to_import)} 只股票...")

    for code, name in stocks_to_import:
        # 尝试自动获取行业
        sector = "未知行业"
        if query_tool:
            try:
                # 尝试直接用代码查
                res = query_tool.get_stock_industry(code)
                if res and res != "未知行业":
                    sector = res
            except Exception as e:
                print(f"获取 {name} 行业失败: {e}")

        # 调用 add_stock
        # 默认给个高分 8 分，因为是“焦佬看好的”
        manager.add_stock(
            code=code,
            name=name,
            sector=sector,
            confidence=8, 
            intro="来自同花顺截图自动导入",
            thesis="待补充",
            guidance="待补充"
        )
        time.sleep(0.1) # 稍微停顿一下

    print("\n批量导入完成！")
    manager.list_stocks()

if __name__ == "__main__":
    batch_import()
