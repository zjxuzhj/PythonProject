import argparse
import signal
import sys
import time
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import os as _os

_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from data_ingestion import updateAllStockDataCache as data_updater
from miniqmt_etf.miniqmt_logging_utils import setup_logger


def run_etf_daily_download():
    try:
        import miniqmt_etf.etf_momentum_rotation_qmt as etf_mod
        if getattr(etf_mod, "strategy_logger", None) is None:
            etf_mod.strategy_logger = setup_logger()
        etf_mod.download_daily_data()
    except Exception as e:
        print(f"ETF日线下载执行异常: {str(e)}")


def configure_scheduler(run_now: bool) -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="Asia/Shanghai")

    scheduler.add_job(
        run_etf_daily_download,
        trigger=CronTrigger(hour=15, minute=1, day_of_week='mon-fri'),
        id='etf_daily_data_downloader',
        misfire_grace_time=300,
        coalesce=True
    )
    print("定时任务已启动：每日15:01执行ETF动量轮动日线数据下载")

    scheduler.add_job(
        data_updater.update_all_daily_data,
        trigger=CronTrigger(hour=15, minute=5, day_of_week='mon-fri'),
        id='daily_data_small_downloader',
        misfire_grace_time=300,
        coalesce=True
    )
    print("定时任务已启动：每日15:05执行小市值策略日线数据下载")

    if run_now:
        print("开始立即执行一次数据下载...")
        run_etf_daily_download()
        data_updater.update_all_daily_data()
        print("立即执行完成。")

    return scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-now", action="store_true")
    args = parser.parse_args()

    scheduler = configure_scheduler(run_now=args.run_now)

    def _shutdown(*_args):
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass
        sys.exit(0)

    for sig in ("SIGINT", "SIGTERM"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), _shutdown)

    scheduler.start()
    print(f"数据下载服务已启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()

