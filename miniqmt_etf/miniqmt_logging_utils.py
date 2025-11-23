# utils/logging_utils.py
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

def setup_logger():
    """配置日志记录器"""
    logger = logging.getLogger("QMT_Strategy")
    logger.setLevel(logging.INFO)

    # 创建按天轮转的日志处理器
    log_handler = TimedRotatingFileHandler(
        "qmt_strategy.log",
        when="midnight",
        interval=1,
        backupCount=7
    )

    # 设置日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    log_handler.setFormatter(formatter)

    # 同时输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 设置控制台编码为UTF-8
    try:
        import io
        console_handler.setStream(io.TextIOWrapper(
            sys.stdout.buffer,
            encoding='utf-8',
            errors='replace'
        ))
    except Exception:
        pass  # 回退到默认流

    logger.addHandler(log_handler)
    logger.addHandler(console_handler)
    return logger
import logging