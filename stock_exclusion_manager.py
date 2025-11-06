# -*- coding: utf-8 -*-
# 模块: stock_exclusion_manager
# 创建目的：将股票排除逻辑独立封装，减少策略文件体积，提升复用与可维护性。
# 使用场景：在 first_limit_up_ma5_normal 与 first_limit_up_ma5_normal_scan 等策略/扫描模块中统一调用排除判断；
#          通过 YAML 配置管理名称/题材模式与黑名单代码；支持可选的热加载。
import logging
import os
import threading
from typing import Dict, Set, Optional, Union, Collection

try:
    import yaml
    _YAML_AVAILABLE = True
except Exception:
    _YAML_AVAILABLE = False

# 依赖现有工具中的代码规范化工具（顶层函数）
from getAllStockCsv import convert_stock_code, code_add_prefix


class StockExclusionManager:
    """
    股票排除管理器（单例）。

    用途：集中管理与执行股票排除规则，避免在各业务模块中散落硬编码逻辑。

    主要接口：
    - get_instance(config_path: Optional[str] = None, hot_reload: bool = False) -> "StockExclusionManager":
      获取单例实例，首次调用时可指定配置路径与是否开启热加载。
    - should_exclude(stock_code: str, stock_name: str = "", themes: Optional[Union[str, Collection[str]]] = None) -> bool:
      综合判断是否排除；返回布尔值，并通过日志输出排除原因。themes 支持字符串（如 "白酒;新能源" 或 "白酒,新能源"）或集合/列表/元组。
    - add_exclusion_rule(pattern: str, targets: Optional[Set[str]] = None, reason: str = "") -> None:
      动态新增名称/题材模式规则。
    - add_blacklist_codes(codes: Set[str], reason: str = "手动黑名单") -> None:
      动态新增代码黑名单。
    - set_hot_reload(enabled: bool) -> None / reload_config() -> None:
      控制或手动触发配置热加载。

    配置文件结构（YAML均可）：
    {
      "pattern_rules": [
        {"pattern": "白酒", "targets": ["name", "theme"], "reason": "行业风险"},
        {"pattern": "光伏", "targets": ["theme"], "reason": "同质化"}
      ],
      "blacklist_codes": ["sz002506", "sh600184"]
    }
    """

    _instance_lock = threading.Lock()
    _instance: Optional["StockExclusionManager"] = None

    def __init__(self, config_path: Optional[str] = None, hot_reload: bool = False) -> None:
        self._blacklist_codes: Set[str] = set()
        self._name_patterns: Dict[str, str] = {}
        self._theme_patterns: Dict[str, str] = {}
        self._hot_reload: bool = hot_reload
        self._config_path: Optional[str] = config_path
        self._config_mtime: Optional[float] = None
        self._check_interval_s: float = 5.0
        self._next_check_ts: float = 0.0

        selected = self._select_config_file(config_path)
        self._config_path = selected

        if self._config_path and os.path.exists(self._config_path):
            self.reload_config()
        else:
            logging.info("StockExclusionManager: 未检测到配置文件，使用空白规则。")

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None, hot_reload: bool = False) -> "StockExclusionManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(config_path=config_path, hot_reload=hot_reload)
            return cls._instance

    @classmethod
    def clear_instance_for_test(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def set_hot_reload(self, enabled: bool) -> None:
        self._hot_reload = enabled

    # 兼容旧接口：可选传入检查间隔
    def set_hot_reload(self, enabled: bool, check_interval_s: float = 5.0) -> None:  # type: ignore[override]
        self._hot_reload = enabled
        try:
            self._check_interval_s = float(check_interval_s)
        except Exception:
            pass

    def reload_config(self) -> None:
        if not self._config_path:
            logging.warning("StockExclusionManager: 无配置路径，跳过加载。")
            return

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                if self._config_path.endswith(".yaml") or self._config_path.endswith(".yml"):
                    if not _YAML_AVAILABLE:
                        logging.warning("StockExclusionManager: 未安装 PyYAML，无法读取 YAML，尝试 JSON 回退。")
                        data = json.load(f)
                    else:
                        data = yaml.safe_load(f) or {}
                else:
                    data = json.load(f)

            self._blacklist_codes = set()
            self._name_patterns = {}
            self._theme_patterns = {}

            for code in data.get("blacklist_codes", []):
                norm = self._normalize_code(code)
                if norm:
                    self._blacklist_codes.add(norm)

            for rule in data.get("pattern_rules", []):
                # 支持两种配置风格：
                # 1) {pattern: "白酒", targets: ["name","theme"], reason: "..."}
                # 2) {type: "name", patterns: ["白酒","酿酒"]}
                reason = str(rule.get("reason", "配置规则")).strip()

                pattern = str(rule.get("pattern", "")).strip()
                targets = set(rule.get("targets", [])) if rule.get("targets") else set()
                if pattern and targets:
                    if "name" in targets:
                        self._name_patterns[pattern] = reason
                    if "theme" in targets:
                        self._theme_patterns[pattern] = reason

                rtype = str(rule.get("type", "")).strip().lower()
                patterns = rule.get("patterns", []) or []
                if rtype in {"name", "theme"} and patterns:
                    for p in patterns:
                        p_str = str(p or "").strip()
                        if not p_str:
                            continue
                        if rtype == "name":
                            self._name_patterns[p_str] = reason
                        else:
                            self._theme_patterns[p_str] = reason

            self._config_mtime = os.path.getmtime(self._config_path)
            logging.info(
                "StockExclusionManager: 配置已加载，规则数 name=%d theme=%d 黑名单=%d | from=%s",
                len(self._name_patterns), len(self._theme_patterns), len(self._blacklist_codes), self._config_path,
            )
        except Exception as e:
            logging.exception("StockExclusionManager: 加载配置失败: %s", e)

    def _select_config_file(self, user_path: Optional[str]) -> Optional[str]:
        if user_path:
            return user_path
        yaml_path = os.path.join("config", "stock_exclusions.yaml")
        if _YAML_AVAILABLE:
            # 优先使用 YAML（可读性与注释支持更好）
            if os.path.exists(yaml_path):
                return yaml_path
        return None

    def _check_and_reload_if_needed(self) -> None:
        if not self._hot_reload or not self._config_path:
            return
        try:
            import time as _t
            now = _t.time()
            if self._check_interval_s > 0 and now < self._next_check_ts:
                return
            self._next_check_ts = now + max(0.0, self._check_interval_s)

            current = os.path.getmtime(self._config_path)
            if self._config_mtime is None or current > self._config_mtime:
                logging.info("StockExclusionManager: 检测到配置变更，执行热加载。")
                self.reload_config()
        except Exception:
            # 忽略热加载异常，避免影响业务流程
            pass

    def add_exclusion_rule(self, pattern: str, targets: Optional[Set[str]] = None, reason: str = "") -> None:
        targets = targets or {"name"}
        reason = reason or "临时规则"
        if "name" in targets:
            self._name_patterns[pattern] = reason
        if "theme" in targets:
            self._theme_patterns[pattern] = reason

    def add_blacklist_codes(self, codes: Set[str], reason: str = "手动黑名单") -> None:
        for c in codes:
            norm = self._normalize_code(c)
            if norm:
                self._blacklist_codes.add(norm)
        logging.info("StockExclusionManager: 动态新增黑名单 %d 条 | 原因: %s", len(codes), reason)

    def should_exclude(self, stock_code: str, stock_name: str = "", themes: Optional[Union[str, Collection[str]]] = None) -> bool:
        self._check_and_reload_if_needed()

        norm_code = self._normalize_code(stock_code)
        if norm_code and norm_code in self._blacklist_codes:
            logging.info(f"排除 {norm_code}：命中代码黑名单")
            return True

        s_name = (stock_name or "").strip()
        for patt, rsn in self._name_patterns.items():
            if patt and s_name and patt in s_name:
                logging.info(f"排除 {norm_code}：名称匹配模式 '{patt}' ({rsn})")
                return True

        # 题材匹配：兼容传入字符串或集合/列表
        for patt, rsn in self._theme_patterns.items():
            if not patt:
                continue
            if isinstance(themes, (set, list, tuple)):
                items = [str(t or "") for t in themes]
                if any(patt in it for it in items):
                    logging.info(f"排除 {norm_code}：题材匹配模式 '{patt}' ({rsn})")
                    return True
            else:
                th = str(themes or "")
                if patt in th:
                    logging.info(f"排除 {norm_code}：题材匹配模式 '{patt}' ({rsn})")
                    return True

        return False

    def _normalize_code(self, code: str) -> Optional[str]:
        try:
            if not code:
                return None
            c = str(code).strip()
            # 标准格式 600138.SH -> sh600138
            if '.' in c:
                try:
                    return convert_stock_code(c)
                except Exception:
                    pass
            # 纯数字 600138 -> sh600138 / 002506 -> sz002506
            if c.isdigit() and len(c) == 6:
                try:
                    return code_add_prefix(c)
                except Exception:
                    pass
            # 已含前缀 sh600138 / SZ002506
            if len(c) > 2 and c[:2].isalpha():
                prefix = c[:2].lower()
                digits = ''.join(ch for ch in c if ch.isdigit())
                return f"{prefix}{digits}"
            return c.lower()
        except Exception:
            return str(code).lower()