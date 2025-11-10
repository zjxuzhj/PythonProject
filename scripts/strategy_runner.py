"""
strategy_runner.py

主控程序：并行管理并执行 miniqmt_base 与 etf_momentum_rotation_qmt 两个实盘策略的执行层。

功能要点：
- 并行运行两个策略进程并实现守护重启、日志分流
- 启动时按总资金与分配比例计算各策略分配资金，并通过环境变量传入子进程
- 支持每日开盘前按比例重新分配（通过重启子进程实现简洁的配置刷新）
- 进程守护：自动重启（最多 N 次），日志采集，异常时告警
- 简易单元测试：包含资金分配计算函数（见 tests/）

注意：策略的核心算法不做改动，脚本通过环境变量覆盖预算常量（若策略在导入时读取这些变量）。
"""

from __future__ import annotations

import os
import sys
import json
import time
import signal
import threading
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PYTHON = sys.executable


def compute_allocations(total_capital: float, allocations: Dict[str, float], max_positions: Dict[str, int]) -> Dict[str, Dict]:
    """按总资金和分配比计算每个策略的资金与每只标的预算。

    返回字典：{strategy: {'allocated': xxx, 'per_stock_budget': yyy, 'max_positions': n}}
    """
    if total_capital <= 0:
        raise ValueError("total_capital must be positive")
    out = {}
    for sname, pct in allocations.items():
        alloc = float(total_capital) * float(pct)
        slots = int(max_positions.get(sname, 1))
        if slots <= 0:
            raise ValueError(f"max_positions for {sname} must be > 0")
        per_stock = alloc / slots
        out[sname] = {'allocated': alloc, 'per_stock_budget': per_stock, 'max_positions': slots}
    return out


class StrategyProcess:
    def __init__(self, name: str, script_path: str, env: Dict[str, str], stdout_path: str, stderr_path: str):
        self.name = name
        self.script_path = script_path
        self.env = os.environ.copy()
        self.env.update(env or {})
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.proc: Optional[subprocess.Popen] = None
        self.restart_attempts = 0
        self.max_restarts = 3

    def start(self):
        cmd = [PYTHON, '-u', self.script_path]
        out_f = open(self.stdout_path, 'a', encoding='utf-8')
        err_f = open(self.stderr_path, 'a', encoding='utf-8')
        self.proc = subprocess.Popen(cmd, env=self.env, stdout=out_f, stderr=err_f, cwd=ROOT)
        self.restart_attempts = 0
        print(f"[{self.name}] started pid={self.proc.pid}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

    def ensure_running(self):
        if self.proc is None or self.proc.poll() is not None:
            self.restart_attempts += 1
            if self.restart_attempts > self.max_restarts:
                print(f"[{self.name}] reached max restart attempts ({self.max_restarts}), not restarting automatically")
                return False
            print(f"[{self.name}] restarting (attempt {self.restart_attempts})...")
            try:
                self.start()
                return True
            except Exception as e:
                print(f"[{self.name}] restart failed: {e}")
                return False
        return True


class StrategyRunner:
    def __init__(self, total_capital: float, allocations: Dict[str, float], max_positions: Dict[str, int], work_dir: str = ROOT):
        self.total_capital = float(total_capital)
        self.allocations = allocations
        self.max_positions = max_positions
        self.work_dir = work_dir
        self.strategy_procs: Dict[str, StrategyProcess] = {}
        self.monitor_thread = None
        self._stop_event = threading.Event()

        # strategy -> script mapping (local project paths)
        self.scripts = {
            'miniqmt': os.path.join(ROOT, 'miniqmt_base.py'),
            'etf': os.path.join(ROOT, 'etf_momentum_rotation_qmt.py')
        }

    def prepare_and_start(self):
        allocs = compute_allocations(self.total_capital, self.allocations, self.max_positions)

        for sname, v in allocs.items():
            script = self.scripts.get(sname)
            if not script or not os.path.exists(script):
                print(f"Script for {sname} not found: {script}")
                continue
            env = {
                # 传入已计算的分配资金（策略内部需读取这些环境变量以覆盖默认常量）
                'STRATEGY_NAME': sname.upper(),
                'ALLOCATED_CAPITAL': str(v['allocated']),
                'PER_STOCK_TOTAL_BUDGET': str(v['per_stock_budget']),
                'MAX_POSITIONS': str(v['max_positions'])
            }
            stdout = os.path.join(self.work_dir, 'analysis_results', f'{sname}.stdout.log')
            stderr = os.path.join(self.work_dir, 'analysis_results', f'{sname}.stderr.log')
            proc = StrategyProcess(sname, script, env, stdout, stderr)
            proc.start()
            self.strategy_procs[sname] = proc

        # start monitor
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _monitor_loop(self):
        print("StrategyRunner monitor started")
        while not self._stop_event.is_set():
            for name, proc in list(self.strategy_procs.items()):
                if not proc.ensure_running():
                    # reached max restarts; alert and continue
                    print(f"[{name}] not running and max restarts reached")
                else:
                    # optional: inspect log files for network errors and restart
                    pass
            time.sleep(10)

    def stop_all(self):
        self._stop_event.set()
        for proc in self.strategy_procs.values():
            proc.stop()

    def daily_reallocate(self, new_total: Optional[float] = None):
        """在每日开盘前按比例重新分配资金（实现为重启子进程以更新 env 变量）。"""
        if new_total:
            self.total_capital = float(new_total)
        print("Daily reallocation: recomputing allocations and restarting strategy processes")
        allocs = compute_allocations(self.total_capital, self.allocations, self.max_positions)
        # restart each process with updated ENV
        for sname, proc in self.strategy_procs.items():
            v = allocs.get(sname)
            if not v:
                continue
            proc.stop()
            proc.env.update({
                'ALLOCATED_CAPITAL': str(v['allocated']),
                'PER_STOCK_TOTAL_BUDGET': str(v['per_stock_budget']),
                'MAX_POSITIONS': str(v['max_positions'])
            })
            proc.start()


def main():
    # 默认配置（可由外部配置文件或命令行注入）
    TOTAL_CAPITAL = float(os.environ.get('TOTAL_CAPITAL', '180000'))
    ALLOC = {
        'miniqmt': float(os.environ.get('ALLOC_MINIQMT', '0.45')),
        'etf': float(os.environ.get('ALLOC_ETF', '0.55'))
    }
    MAX_POS = {
        'miniqmt': int(os.environ.get('MAX_POS_MINIQMT', '10')),
        'etf': int(os.environ.get('MAX_POS_ETF', '5'))
    }

    runner = StrategyRunner(TOTAL_CAPITAL, ALLOC, MAX_POS, work_dir=ROOT)
    try:
        runner.prepare_and_start()

        # 简单的主循环：支持每日 08:50 的重分配（本示例用睡眠替代真正的调度器）
        while True:
            now = datetime.now()
            # 每日 08:50 触发重分配一次（仅示例，可修改）
            if now.hour == 8 and now.minute == 50:
                runner.daily_reallocate()
                # sleep 61s to avoid重复触发
                time.sleep(61)
            time.sleep(5)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping strategies...")
    finally:
        runner.stop_all()


if __name__ == '__main__':
    main()