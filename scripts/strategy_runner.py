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

def fetch_total_capital_from_qmt() -> Optional[float]:
    """尝试通过 miniQMT 读取账户总资金，参考 miniqmt_base.py 的资产查询。"""
    try:
        from xtquant import xtdata
        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount
        xtdata.enable_hello = False
        user_dir = os.environ.get('QMT_USERDATA_DIR') or r'D:\备份\国金证券QMT交易端\userdata_mini'
        account_id = os.environ.get('QMT_ACCOUNT_ID') or '8886969255'
        session_id = int(time.time())
        trader = XtQuantTrader(user_dir, session_id)
        trader.start()
        try:
            trader.connect()
        except Exception:
            pass
        acc = StockAccount(account_id, 'STOCK')
        asset_val: Optional[float] = None
        asset = trader.query_stock_asset(acc)
        if asset:
            for field in ('m_dTotalAsset', 'm_dTotalCapital', 'm_dTotalEquity', 'm_dCash'):
                val = getattr(asset, field, None)
                try:
                    if val and float(val) > 0:
                        asset_val = float(val)
                        break
                except Exception:
                    continue
        # 主动停止 trader 线程，避免长连接占用资源
        try:
            trader.stop()
        except Exception:
            pass
        return asset_val
    except Exception:
        return None


def _tail_file(path: str, n_lines: int = 10) -> str:
    """读取文件末尾若干行，优先用 UTF-8 解码，失败时回退 GBK/CP936，避免中文乱码。"""
    try:
        if not path or not os.path.exists(path):
            return "<no file>"
        with open(path, 'rb') as f:
            lines = f.readlines()
        tail = lines[-n_lines:] if len(lines) > n_lines else lines
        raw = b''.join(tail)
        try:
            return raw.decode('utf-8').strip()
        except UnicodeDecodeError:
            try:
                return raw.decode('gbk').strip()
            except Exception:
                return raw.decode('utf-8', errors='replace').strip()
    except Exception as e:
        return f"<tail error: {e}>"


def _read_new(path: str, offsets: dict, max_bytes: int = 8192) -> str:
    try:
        if not path or not os.path.exists(path):
            return ""
        size = os.path.getsize(path)
        last = int(offsets.get(path, 0) or 0)
        if last > size:
            last = 0
        with open(path, 'rb') as f:
            f.seek(last)
            data = f.read(max(0, size - last))
        offsets[path] = last + len(data)
        if not data:
            return ""
        try:
            s = data.decode('utf-8').strip()
        except UnicodeDecodeError:
            try:
                s = data.decode('gbk').strip()
            except Exception:
                s = data.decode('utf-8', errors='replace').strip()
        if len(s) > max_bytes:
            return s[-max_bytes:]
        return s
    except Exception:
        return ""


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
        self.start_time: Optional[float] = None

    def start(self):
        cmd = [PYTHON, '-u', self.script_path]
        out_dir = os.path.dirname(self.stdout_path)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        out_f = open(self.stdout_path, 'a', encoding='utf-8')
        err_f = open(self.stderr_path, 'a', encoding='utf-8')
        # 强制子进程 stdout/stderr 使用 UTF-8，避免中文乱码
        self.env['PYTHONIOENCODING'] = 'utf-8'
        self.proc = subprocess.Popen(cmd, env=self.env, stdout=out_f, stderr=err_f, cwd=ROOT)
        self.restart_attempts = 0
        self.start_time = time.time()
        # 打印环境覆盖摘要，帮助核对分配参数是否正确
        try:
            env_summary = {
                'ALLOCATED_CAPITAL': self.env.get('ALLOCATED_CAPITAL'),
                'PER_STOCK_TOTAL_BUDGET': self.env.get('PER_STOCK_TOTAL_BUDGET'),
                'MAX_POSITIONS': self.env.get('MAX_POSITIONS'),
                # 'PYTHONIOENCODING': self.env.get('PYTHONIOENCODING'),
            }
            print(f"[{self.name}] started pid={self.proc.pid} env={env_summary}")
        except Exception:
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
            # 退避重试，避免快速重启导致频繁失败
            backoff = min(30, self.restart_attempts * 5)
            if backoff > 0:
                print(f"[{self.name}] backoff {backoff}s before restart")
                time.sleep(backoff)
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
        self._log_offsets = {}

        # strategy -> script mapping (local project paths)
        self.scripts = {
            'miniqmt': os.path.join(ROOT, 'miniqmt_etf', 'miniqmt_base.py'),
            'etf': os.path.join(ROOT, 'miniqmt_etf', 'etf_momentum_rotation_qmt.py')
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
            try:
                self._log_offsets[stdout] = os.path.getsize(stdout) if os.path.exists(stdout) else 0
                self._log_offsets[stderr] = os.path.getsize(stderr) if os.path.exists(stderr) else 0
            except Exception:
                self._log_offsets[stdout] = 0
                self._log_offsets[stderr] = 0

        # start monitor
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _monitor_loop(self):
        # print("StrategyRunner monitor started (heartbeat every ~10s)")
        while not self._stop_event.is_set():
            try:
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # print(f"[monitor] {ts} checking child processes...")
                for name, proc in list(self.strategy_procs.items()):
                    status = 'running' if (proc.proc and proc.proc.poll() is None) else f"stopped(exit={proc.proc.poll() if proc.proc else 'N/A'})"
                    if status.startswith('stopped'):
                        ok = proc.ensure_running()
                        print(f"[monitor] {name} restart_attempts={proc.restart_attempts} restarted={ok}")
                    new_out = _read_new(proc.stdout_path, self._log_offsets)
                    new_err = _read_new(proc.stderr_path, self._log_offsets)
                    if new_out:
                        print(f"[monitor] {name} stdout new:\n{new_out}")
                    if new_err:
                        print(f"[monitor] {name} stderr new:\n{new_err}")
                    if new_out and '程序退出' in new_out:
                        print(f"[monitor] {name} detected exit marker '程序退出' in stdout")
            except Exception as e:
                print(f"[monitor] error: {e}")
            time.sleep(10)

    def stop_all(self):
        self._stop_event.set()
        for proc in self.strategy_procs.values():
            proc.stop()

    def daily_reallocate(self, new_total: Optional[float] = None):
        """在每日开盘前按比例重新分配资金（实现为重启子进程以更新 env 变量）。"""
        if new_total is None:
            # 优先尝试从QMT读取最新总资金
            fresh = fetch_total_capital_from_qmt()
            if fresh is not None and fresh > 0:
                print(f"[reallocate] fetched total capital from QMT: {fresh:,.2f}")
                self.total_capital = float(fresh)
        else:
            self.total_capital = float(new_total)
        print(f"[reallocate] recomputing allocations with total_capital={self.total_capital:,.2f}")
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
            print(f"[reallocate] {sname} new env={{'ALLOCATED_CAPITAL': {v['allocated']}, 'PER_STOCK_TOTAL_BUDGET': {v['per_stock_budget']}, 'MAX_POSITIONS': {v['max_positions']}}}")
            proc.start()


def main():
    # 默认配置（可由外部配置文件或命令行注入）
    qc = fetch_total_capital_from_qmt()
    if qc is not None and qc > 0:
        TOTAL_CAPITAL = float(qc)
        print(f"[strategy_runner] 从 miniQMT 读取账户总资金: {TOTAL_CAPITAL:,.2f}")
    else:
        TOTAL_CAPITAL = float(os.environ.get('TOTAL_CAPITAL', '180000'))
        print(f"[strategy_runner] 使用默认/环境总资金: {TOTAL_CAPITAL:,.2f}")
    ALLOC = {
        'miniqmt': float(os.environ.get('ALLOC_MINIQMT', '0.65')),
        'etf': float(os.environ.get('ALLOC_ETF', '0.35'))
    }
    MAX_POS = {
        'miniqmt': int(os.environ.get('MAX_POS_MINIQMT', '5')),
        'etf': int(os.environ.get('MAX_POS_ETF', '1'))
    }

    scripts_dir = os.path.abspath(os.path.dirname(__file__))
    runner = StrategyRunner(TOTAL_CAPITAL, ALLOC, MAX_POS, work_dir=scripts_dir)
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