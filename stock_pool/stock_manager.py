import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional

# 添加上级目录到系统路径，以便导入 getAllStockCsv
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from getAllStockCsv import StockQuery, code_add_prefix
    STOCK_QUERY_AVAILABLE = True
except ImportError:
    STOCK_QUERY_AVAILABLE = False
    print("警告: 无法导入 getAllStockCsv，自动填充功能将不可用。")

class StockPoolManager:
    def __init__(self, filepath: str = 'stock_pool.json'):
        self.filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_pool.json')
        self.stocks: List[Dict] = []
        self._load_data()

    def _load_data(self):
        """加载 JSON 数据"""
        if not os.path.exists(self.filepath):
            self.stocks = []
            return
        
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self.stocks = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self.stocks = []

    def _save_data(self):
        """保存数据到 JSON"""
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.stocks, f, ensure_ascii=False, indent=4)
            print(f"数据已保存至 {self.filepath}")
        except Exception as e:
            print(f"保存失败: {e}")

    def add_stock(self, code: str, name: str, sector: str, confidence: int, 
                  intro: str, thesis: str, guidance: str, recommendation_time: str = ""):
        """添加新股票"""
        # 检查是否已存在
        for stock in self.stocks:
            if stock['code'] == code:
                print(f"错误: 股票代码 {code} ({stock['name']}) 已存在！")
                return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_stock = {
            "code": code,
            "name": name,
            "sector": sector,
            "confidence": confidence,
            "intro": intro,
            "thesis": thesis,
            "guidance": guidance,
            "recommendation_time": recommendation_time,
            "created_at": now,
            "updated_at": now
        }
        self.stocks.append(new_stock)
        self._save_data()
        print(f"成功添加股票: {name} ({code})")

    def remove_stock(self, code: str):
        """移除股票"""
        original_count = len(self.stocks)
        self.stocks = [s for s in self.stocks if s['code'] != code]
        if len(self.stocks) < original_count:
            self._save_data()
            print(f"成功移除股票: {code}")
        else:
            print(f"未找到股票代码: {code}")

    def update_stock(self, code: str, **kwargs):
        """更新股票信息"""
        found = False
        for stock in self.stocks:
            if stock['code'] == code:
                found = True
                # 更新允许的字段
                allowed_fields = ['name', 'sector', 'confidence', 'intro', 'thesis', 'guidance', 'recommendation_time']
                updated = False
                for key, value in kwargs.items():
                    if key in allowed_fields and value is not None and value != "":
                        stock[key] = value
                        updated = True
                
                if updated:
                    stock['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self._save_data()
                    print(f"成功更新股票: {stock['name']} ({code})")
                else:
                    print("没有字段被更新。")
                break
        
        if not found:
            print(f"未找到股票代码: {code}")

    def list_stocks(self, sort_by: str = 'confidence'):
        """列出所有股票"""
        if not self.stocks:
            print("股票池为空。")
            return

        # 排序
        reverse = True # 默认降序（看好程度）
        if sort_by not in ['confidence', 'code']:
            sort_by = 'confidence'
        
        sorted_stocks = sorted(self.stocks, key=lambda x: x.get(sort_by, 0), reverse=reverse)

        print(f"\n{'='*80}")
        print(f"{'代码':<8} {'名称':<10} {'行业':<8} {'看好(1-10)':<10} {'操作指导'}")
        print(f"{'-'*80}")
        for s in sorted_stocks:
            print(f"{s['code']:<8} {s['name']:<10} {s['sector']:<8} {s['confidence']:<10} {s['guidance'][:20]}...")
        print(f"{'='*80}\n")

    def get_stock_detail(self, code: str):
        """获取单个股票详情"""
        for s in self.stocks:
            if s['code'] == code:
                print(f"\n{'='*40}")
                print(f"股票: {s['name']} ({s['code']})")
                print(f"行业: {s['sector']}")
                print(f"看好程度: {s['confidence']}/10")
                print(f"{'-'*40}")
                print(f"【简介】: {s['intro']}")
                print(f"【核心逻辑】: {s['thesis']}")
                print(f"【操作指导】: {s['guidance']}")
                print(f"更新时间: {s['updated_at']}")
                print(f"{'='*40}\n")
                return
        print(f"未找到股票代码: {code}")

# 交互式命令行工具
def input_stock_data(query_tool=None):
    print("\n--- 录入新股票 ---")
    
    code = input("代码 (如 600519): ").strip()
    
    default_name = ""
    default_sector = ""
    
    if query_tool and code.isdigit() and len(code) == 6:
        # 获取名称
        try:
            # 尝试添加前缀获取名称（StockQuery 可能需要前缀）
            prefixed_code = code_add_prefix(code)
            name_result = query_tool.get_name_by_code(prefixed_code)
            if name_result and name_result != "未找到对应名称":
                default_name = name_result
        except Exception as e:
            print(f"获取名称失败: {e}")
            
        # 获取行业
        try:
            # StockQuery 的 get_stock_industry 通常使用无前缀代码作为 key（根据 industry_cache 初始化逻辑）
            # 或者尝试两种方式
            sector_result = query_tool.get_stock_industry(code)
            if sector_result and sector_result != "未知行业":
                default_sector = sector_result
        except Exception as e:
            print(f"获取行业失败: {e}")

    prompt_name = f"名称 (默认为 {default_name}): " if default_name else "名称 (如 贵州茅台): "
    name = input(prompt_name).strip()
    if not name and default_name:
        name = default_name

    prompt_sector = f"板块/行业 (默认为 {default_sector}): " if default_sector else "板块/行业 (如 白酒): "
    sector = input(prompt_sector).strip()
    if not sector and default_sector:
        sector = default_sector
    
    while True:
        try:
            confidence = int(input("看好程度 (1-10): ").strip())
            if 1 <= confidence <= 10:
                break
            print("请输入 1 到 10 之间的整数。")
        except ValueError:
            print("请输入有效的整数。")
            
    intro = input("简介 (一句话描述): ").strip()
    thesis = input("核心逻辑 (为什么看好?): ").strip()
    guidance = input("操作指导 (买入/卖出/持有建议): ").strip()
    
    return code, name, sector, confidence, intro, thesis, guidance

def main():
    manager = StockPoolManager()
    
    # 全局初始化一次 StockQuery，避免每次调用都重新加载
    global StockQuery, code_add_prefix
    query_tool_instance = None
    if STOCK_QUERY_AVAILABLE:
        try:
            print("正在初始化股票数据，请稍候...")
            query_tool_instance = StockQuery()
        except Exception as e:
            print(f"StockQuery 初始化失败: {e}")

    while True:
        print("\n=== 股票池管理系统 ===")
        print("1. 查看股票池 (List)")
        print("2. 添加股票 (Add)")
        print("3. 查看详情 (Detail)")
        print("4. 更新股票 (Update)")
        print("5. 删除股票 (Remove)")
        print("6. 退出 (Exit)")
        
        choice = input("请选择操作 (1-6): ").strip()
        
        if choice == '1':
            sort_key = input("排序方式 (c:看好程度, 默认c): ").strip().lower()
            manager.list_stocks(sort_by='confidence')
            
        elif choice == '2':
            data = input_stock_data(query_tool_instance)
            manager.add_stock(*data)
            
        elif choice == '3':
            code = input("请输入要查看的股票代码: ").strip()
            manager.get_stock_detail(code)

        elif choice == '4':
            code = input("请输入要更新的股票代码: ").strip()
            print("直接回车跳过不修改的字段")
            confidence_str = input("新看好程度 (1-10): ").strip()
            thesis = input("新核心逻辑: ").strip()
            guidance = input("新操作指导: ").strip()
            
            updates = {}
            if confidence_str: updates['confidence'] = int(confidence_str)
            if thesis: updates['thesis'] = thesis
            if guidance: updates['guidance'] = guidance
            
            if updates:
                manager.update_stock(code, **updates)
            else:
                print("未输入任何更新内容。")

        elif choice == '5':
            code = input("请输入要删除的股票代码: ").strip()
            confirm = input(f"确定要删除 {code} 吗? (y/n): ").strip().lower()
            if confirm == 'y':
                manager.remove_stock(code)
                
        elif choice == '6':
            print("退出系统。再见！")
            break
        else:
            print("无效输入，请重试。")

if __name__ == '__main__':
    main()
