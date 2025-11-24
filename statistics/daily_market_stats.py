import argparse
import logging
import sys
import re
from pathlib import Path
from datetime import datetime, date

try:
    import pandas as pd
except ImportError:
    pd = None


def parse_date_str(s: str) -> date:
    s = s.strip()
    if re.match(r"^\d{8}$", s):
        return datetime.strptime(s, "%Y%m%d").date()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return datetime.strptime(s, "%Y-%m-%d").date()
    raise ValueError(f"无效日期格式: {s}")


def format_date(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def collect_symbol_files(data_cache: Path) -> dict[str, Path]:
    pattern = re.compile(r"^stock_(.+)_(\d{8})\.parquet$")
    by_symbol: dict[str, Path] = {}
    for p in data_cache.glob("stock_*_*.parquet"):
        m = pattern.match(p.name)
        if not m:
            continue
        symbol = m.group(1)
        by_symbol[symbol] = p
    return by_symbol


def detect_return_column(df: "pd.DataFrame") -> str | None:
    candidates = [
        "pct_chg",
        "pct_change",
        "change_pct",
        "returns",
        "return",
        "chg_pct",
        "ret",
        "涨跌幅",
        "涨跌率",
        "涨幅",
        "跌幅",
        "涨跌幅(%)",
    ]
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {str(c).strip().lower(): c for c in cols}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    for col in cols:
        lc = str(col).lower()
        if ("pct" in lc and ("chg" in lc or "change" in lc)) or ("return" in lc or "ret" in lc):
            return col
        if "涨跌" in col or "涨幅" in col or "跌幅" in col:
            return col
    return None


def compute_symbol_return(df: "pd.DataFrame") -> float | None:
    col = detect_return_column(df)
    if col is not None:
        s = df[col]
        if s.dtype == object:
            s = s.astype(str).str.replace("%", "", regex=False)
        s = pd.to_numeric(s, errors="coerce")
        s = s.dropna()
        if s.empty:
            return None
        v = s.iloc[-1]
        fv = float(v)
        if abs(fv) > 1 and abs(fv) <= 100:
            return fv / 100.0
        return fv
    if "close" in df.columns and "pre_close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce").iloc[-1]
        pre_close = pd.to_numeric(df["pre_close"], errors="coerce").iloc[-1]
        if pd.isna(close) or pd.isna(pre_close) or pre_close == 0:
            return None
        return float((close - pre_close) / pre_close)
    return None


def detect_date_column(df: "pd.DataFrame") -> str | None:
    candidates = [
        "date",
        "trade_date",
        "datetime",
        "timestamp",
        "日期",
        "交易日期",
        "时间",
        "交易时间",
    ]
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {str(c).strip().lower(): c for c in cols}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def extract_returns_by_date(df: "pd.DataFrame") -> dict[date, float]:
    col = detect_return_column(df)
    if col is None:
        return {}
    s = df[col]
    if s.dtype == object:
        s = s.astype(str).str.replace("%", "", regex=False)
    s = pd.to_numeric(s, errors="coerce")
    date_col = detect_date_column(df)
    if date_col is not None:
        dser = pd.to_datetime(df[date_col], errors="coerce")
    else:
        try:
            dser = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            dser = pd.Series([pd.NaT] * len(df))
    out_df = pd.DataFrame({"dt": dser, "ret": s})
    out_df = out_df.dropna(subset=["dt", "ret"])  # type: ignore
    if out_df.empty:
        return {}
    out_df["date"] = out_df["dt"].dt.date
    grp = out_df.groupby("date")["ret"].last()
    res: dict[date, float] = {}
    for d, v in grp.items():
        fv = float(v)
        if abs(fv) > 1 and abs(fv) <= 100:
            fv = fv / 100.0
        res[d] = fv
    return res

def get_return_for_date(df: "pd.DataFrame", d: date) -> float | None:
    col = detect_return_column(df)
    if col is None:
        return None
    try:
        idx = pd.to_datetime(df.index, errors="coerce")
    except Exception:
        return None
    if idx.isna().all():
        return None
    mask = (idx.date == d)
    if not mask.any():
        return None
    rows = df.loc[mask]
    if isinstance(rows, pd.Series):
        s = rows[col]
        if isinstance(s, str):
            s = s.replace("%", "")
        v = pd.to_numeric(s, errors="coerce")
        if pd.isna(v):
            return None
        fv = float(v)
        if abs(fv) > 1 and abs(fv) <= 100:
            fv = fv / 100.0
        return fv
    else:
        ser = rows[col]
        if ser.dtype == object:
            ser = ser.astype(str).str.replace("%", "", regex=False)
        ser = pd.to_numeric(ser, errors="coerce").dropna()
        if ser.empty:
            return None
        fv = float(ser.iloc[-1])
        if abs(fv) > 1 and abs(fv) <= 100:
            fv = fv / 100.0
        return fv


def process_dates(date_list: list[date], by_symbol: dict[str, Path], logger: logging.Logger) -> dict[date, tuple[int, int]]:
    counts: dict[date, tuple[int, int]] = {d: (0, 0) for d in date_list}
    total = len(by_symbol)
    processed = 0
    for symbol, path in by_symbol.items():
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"读取失败: {path} ({symbol}) {e}")
            continue
        col = detect_return_column(df)
        if col is None:
            processed += 1
            if processed % 500 == 0:
                logger.info(f"已处理 {processed}/{total}")
            continue
        ser = df[col]
        if ser.dtype == object:
            ser = ser.astype(str).str.replace("%", "", regex=False)
        ser = pd.to_numeric(ser, errors="coerce")
        try:
            idx = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            processed += 1
            if processed % 500 == 0:
                logger.info(f"已处理 {processed}/{total}")
            continue
        ok = ~idx.isna() & ~ser.isna()
        out_df = pd.DataFrame({"dt": idx[ok], "ret": ser[ok]})
        if not out_df.empty:
            out_df["date"] = out_df["dt"].dt.date
            grp = out_df.groupby("date")["ret"].last()
            for d in date_list:
                if d not in grp.index:
                    continue
                val = float(grp.loc[d])
                if abs(val) > 1 and abs(val) <= 100:
                    val = val / 100.0
                up, down = counts[d]
                if val > 0:
                    up += 1
                else:
                    down += 1
                counts[d] = (up, down)
        processed += 1
        if processed % 500 == 0:
            logger.info(f"已处理 {processed}/{total}")
    for d in date_list:
        up, down = counts[d]
        logger.info(f"{format_date(d)} 统计完成: 上涨={up}, 下跌={down}, 样本={total}")
    return counts


def load_existing(csv_path: Path, logger: logging.Logger) -> "pd.DataFrame":
    if not csv_path.exists():
        return pd.DataFrame(columns=["date", "up_count", "down_count"], dtype="object")
    try:
        df = pd.read_csv(csv_path, dtype={"date": "string", "up_count": "Int64", "down_count": "Int64"})
    except Exception as e:
        logger.error(f"读取现有CSV失败: {csv_path} {e}")
        raise
    return df


def write_csv(csv_path: Path, df: "pd.DataFrame") -> None:
    df = df.copy()
    df["date"] = df["date"].astype("string")
    df["up_count"] = pd.to_numeric(df["up_count"], errors="coerce").astype("Int64")
    df["down_count"] = pd.to_numeric(df["down_count"], errors="coerce").astype("Int64")
    df = df.sort_values("date")
    df.to_csv(csv_path, index=False)


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("daily_market_stats")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(output_dir / "daily_market_stats.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    return logger


def main():
    parser = argparse.ArgumentParser(description="每日股票涨跌家数统计并输出CSV")
    parser.add_argument("--data-cache", type=str, default=str(Path.cwd() / "data_cache"), help="data_cache目录路径")
    parser.add_argument("--start-date", type=str, default=None, help="开始日期，YYYY-MM-DD或YYYYMMDD")
    parser.add_argument("--end-date", type=str, default=None, help="结束日期，YYYY-MM-DD或YYYYMMDD")
    parser.add_argument("--overwrite-dates", type=str, default=None, help="强制覆盖的日期，逗号分隔")
    parser.add_argument("--overwrite-range", nargs=2, type=str, default=None, help="强制覆盖的日期范围 起 始")
    parser.add_argument("--list-missing", action="store_true", help="仅列出指定日期范围内无有效涨跌幅的股票列表")
    args = parser.parse_args()

    data_cache = Path(args.data_cache).resolve()
    if not data_cache.exists():
        print(f"data_cache不存在: {data_cache}")
        sys.exit(1)

    output_dir = data_cache.parent / "statistics"
    logger = setup_logger(output_dir)

    if pd is None:
        logger.error("未安装pandas，请安装pandas及pyarrow/fastparquet以读取parquet文件")
        sys.exit(1)

    sd = parse_date_str(args.start_date) if args.start_date else None
    ed = parse_date_str(args.end_date) if args.end_date else None
    if sd and ed and sd > ed:
        logger.error("开始日期不能晚于结束日期")
        sys.exit(1)

    overwrite_dates: set[date] = set()
    if args.overwrite_dates:
        for s in args.overwrite_dates.split(","):
            overwrite_dates.add(parse_date_str(s))
    if args.overwrite_range:
        od1 = parse_date_str(args.overwrite_range[0])
        od2 = parse_date_str(args.overwrite_range[1])
        if od1 > od2:
            od1, od2 = od2, od1
        cur = od1
        while cur <= od2:
            overwrite_dates.add(cur)
            cur = (datetime.combine(cur, datetime.min.time()) + pd.Timedelta(days=1)).date()

    by_symbol = collect_symbol_files(data_cache)
    if not by_symbol:
        logger.warning("未发现匹配的parquet文件")
    if args.list_missing:
        if sd is None or ed is None:
            logger.error("列出缺失需要提供开始与结束日期")
            return
        date_list = [d.date() for d in pd.date_range(sd, ed, freq="D")]
        total = len(by_symbol)
        for d in date_list:
            missing = []
            for symbol, path in by_symbol.items():
                try:
                    df = pd.read_parquet(path)
                    ret_map = extract_returns_by_date(df)
                    r = ret_map.get(d)
                except Exception:
                    r = None
                if r is None:
                    missing.append((symbol, str(path)))
            logger.info(f"{format_date(d)} 无有效涨跌幅: {len(missing)}/{total}")
            if missing:
                out_csv = output_dir / f"missing_returns_{format_date(d)}.csv"
                mdf = pd.DataFrame(missing, columns=["symbol", "file"])
                mdf.to_csv(out_csv, index=False)
                logger.info(f"缺失列表已写出: {out_csv}")
        return

    csv_path = output_dir / "daily_market_stats.csv"
    existing = load_existing(csv_path, logger)
    existing_dates = set()
    if not existing.empty and "date" in existing.columns:
        try:
            existing_dates = set(datetime.strptime(x, "%Y-%m-%d").date() for x in existing["date"].dropna().astype(str))
        except Exception:
            existing_dates = set()

    if sd is None or ed is None:
        logger.error("请提供开始与结束日期以进行统计")
        sys.exit(1)
    date_list = [d.date() for d in pd.date_range(sd, ed, freq="D")]
    rows = []
    counts = process_dates(date_list, by_symbol, logger)
    for d in date_list:
        if d in existing_dates and d not in overwrite_dates:
            continue
        up, down = counts.get(d, (0, 0))
        rows.append({"date": format_date(d), "up_count": int(up), "down_count": int(down)})

    result = existing.copy()
    if rows:
        add_df = pd.DataFrame(rows)
        if not result.empty:
            result = pd.concat([result[~result["date"].isin(add_df["date"])], add_df], ignore_index=True)
        else:
            result = add_df
    write_csv(csv_path, result)
    logger.info(f"写入完成: {csv_path}")


if __name__ == "__main__":
    main()


def update_counts_for_date(data_cache_dir: str | Path, target_date: date, up_count: int, down_count: int, ts: datetime | None = None) -> None:
    data_cache = Path(data_cache_dir).resolve()
    output_dir = data_cache.parent / "statistics"
    logger = setup_logger(output_dir)
    row = {"date": format_date(target_date), "up_count": int(up_count), "down_count": int(down_count)}
    csv_path = output_dir / "daily_market_stats.csv"
    existing = load_existing(csv_path, logger)
    if not existing.empty:
        result = pd.concat([existing[existing["date"] != row["date"]], pd.DataFrame([row])], ignore_index=True)
    else:
        result = pd.DataFrame([row])
    write_csv(csv_path, result)
    if ts is not None:
        logger.info(f"{format_date(target_date)} 统计写入: 上涨={up_count}, 下跌={down_count}, 时间戳={ts.isoformat()}")
