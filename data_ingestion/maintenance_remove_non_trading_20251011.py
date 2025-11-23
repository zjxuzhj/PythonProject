"""
批量清理 data_cache 中通过 get_stock_data 读取的缓存数据，移除 2025-10-11（非交易日）的错误记录。

处理步骤：
- 遍历 data_cache 下的所有 get_stock_data 缓存文件（模式：stock_*_20240201.parquet）
- 若存在索引为 2025-10-11 的记录，则删除该行并回写文件
- 保留其它日期数据不变，并在写回后进行校验
- 特别打印并验证 sz002985 文件的清理结果
"""

import os
import sys
import traceback
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data_cache')
TARGET_DATE = pd.Timestamp('2025-10-11')


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """确保索引为 DatetimeIndex；如果存在 'date' 列，则优先使用其作为索引。"""
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            except Exception:
                pass
    return df


def process_file(fp: str) -> dict:
    before_count = None
    after_count = None
    removed = 0
    success = True
    msg = ''
    try:
        df = pd.read_parquet(fp, engine='fastparquet')
        df = normalize_df(df)
        before_count = len(df)

        if TARGET_DATE in df.index:
            # 删除目标日期并回写
            df = df.drop(index=[TARGET_DATE])
            removed = 1
            df.to_parquet(fp, engine='fastparquet')

            # 验证写回后的文件不包含目标日期
            df2 = pd.read_parquet(fp, engine='fastparquet')
            df2 = normalize_df(df2)
            after_count = len(df2)
            assert TARGET_DATE not in df2.index, '写回后仍存在目标日期'
        else:
            after_count = before_count
    except Exception as e:
        success = False
        msg = f"{e}\n{traceback.format_exc()}"
    return {
        'file': os.path.basename(fp),
        'path': fp,
        'removed': removed,
        'before': before_count,
        'after': after_count,
        'success': success,
        'error': msg,
    }


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"数据目录不存在: {DATA_DIR}")
        sys.exit(1)

    # 仅处理 get_stock_data 约定的文件格式（2024集）
    targets = []
    for name in os.listdir(DATA_DIR):
        if not name.endswith('_20240201.parquet'):
            continue
        if not name.startswith('stock_'):
            continue
        targets.append(os.path.join(DATA_DIR, name))

    print(f"发现 {len(targets)} 个缓存文件需要检查")
    results = []
    for fp in sorted(targets):
        res = process_file(fp)
        results.append(res)
        status = 'OK' if res['success'] else 'FAIL'
        changed = 'REMOVED' if res['removed'] else 'UNCHANGED'
        print(f"[{status}] {changed} {res['file']} | rows: {res['before']} -> {res['after']}")

    total = len(results)
    changed_count = sum(r['removed'] for r in results)
    failed = [r for r in results if not r['success']]
    print(f"\n总结: 处理 {total} 个文件，移除目标日期 {changed_count} 个文件")
    if failed:
        print(f"失败 {len(failed)} 个文件：")
        for r in failed:
            print(f" - {r['file']}: {r['error']}")

    # 特别验证 sz002985
    sz_fp = os.path.join(DATA_DIR, 'stock_sz002985_20240201.parquet')
    if os.path.exists(sz_fp):
        try:
            df = pd.read_parquet(sz_fp, engine='fastparquet')
            df = normalize_df(df)
            has_target = TARGET_DATE in df.index
            print(f"\n[北摩高科 sz002985] 清理验证: 目标日期存在? {has_target}")
        except Exception as e:
            print(f"\n[北摩高科 sz002985] 验证失败: {e}")
    else:
        print("\n[北摩高科 sz002985] 文件不存在，跳过验证")


if __name__ == '__main__':
    main()