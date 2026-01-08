#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""log_mult pipeline v6 (bm 생성 + 48→4 압축)

입력(기본):
  - log_mult_ch.csv
  - log_mult_wgt.csv

동작:
  1) (선택) 기존 bm 파일이 있으면 그걸 사용. 없으면 ch×wgt로 bm 생성
  2) bm을 48행 블록(테이블)들로 나눔 (전체 행수가 48의 배수여야 함)
  3) 모든 테이블을 같은 위치(row, col)끼리 element-wise로 합산 → 합계(48행) 테이블
     - 즉, (table1 1행 + table2 1행 + ...), (table1 2행 + table2 2행 + ...) ...
  4) 합계(48행)를 4행으로 압축 (긴 4줄):
     - out_row0 = sum(rows 1,5,9,...,45)  (0-index 기준: 0,4,8,...,44)
     - out_row1 = sum(rows 2,6,10,...,46) (1,5,9,...,45)
     - out_row2 = sum(rows 3,7,11,...,47)
     - out_row3 = sum(rows 4,8,12,...,48)

출력(기본):
  - log_mult_bm.csv
  - log_mult_bm_sum_48.csv  (옵션 --write_sum)
  - log_mult_final_4.csv

주의:
- bm를 wgt에서 재생성하면, 원본 bm가 더 높은 정밀도로 저장돼 있었던 경우(예: 13자리),
  wgt의 반올림/절삭 때문에 숫자가 '완전히 동일'하게는 안 맞을 수 있음.
- 최종 합산/압축 로직은 테이블 개수가 4든 12든(=48행 블록 개수) 자동으로 전부 합산합니다.

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# CSV parsing helpers
# -----------------------------
def read_csv_str(path: Path) -> pd.DataFrame:
    """Read CSV as strings, preserving empty cells; blank lines are ignored."""
    return pd.read_csv(
        path,
        header=None,
        dtype=str,
        keep_default_na=False,
        skip_blank_lines=True,
        engine="python",
    )


def extract_numeric(df_str: pd.DataFrame) -> Tuple[np.ndarray, int]:
    """
    Return (numeric_matrix, empty_tail_cols)
    - numeric_matrix keeps only columns that are not fully empty.
    - empty_tail_cols counts trailing fully-empty columns (common if rows end with comma).
    """
    df = df_str.copy()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    empty_col = [(df[c] == "").all() for c in df.columns]

    empty_tail = 0
    for flag in reversed(empty_col):
        if flag:
            empty_tail += 1
        else:
            break

    numeric_cols = [c for c, flag in zip(df.columns, empty_col) if not flag]
    if not numeric_cols:
        raise ValueError("No numeric columns found (file appears empty).")

    num = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if num.isna().any().any():
        bad = np.argwhere(num.isna().to_numpy())
        examples = []
        for r, cidx in bad[:8]:
            raw = df.iloc[int(r), int(num.columns[cidx])]
            examples.append((int(r), int(num.columns[cidx]), raw))
        raise ValueError(f"Non-numeric cells found. Examples (row, col, raw): {examples}")

    return num.to_numpy(dtype=float), empty_tail


# -----------------------------
# Complex multiply (ch × wgt)
# -----------------------------
def complex_mul_rowwise(
    ch_reim: np.ndarray,
    wgt_reim: np.ndarray,
    ch_mode: str = "row0",
    ch_row: int = 0,
) -> np.ndarray:
    """
    Multiply complex scalars from ch with each complex pair in wgt.

    ch_reim: (N, >=2) -> uses first two columns as (Re,Im)
    wgt_reim: (N, 2*K) -> [Re0,Im0,Re1,Im1,...]
    Returns: (N, 2*K)
    """
    if wgt_reim.shape[1] % 2 != 0:
        raise ValueError(f"wgt numeric columns must be even (Re/Im pairs). got {wgt_reim.shape[1]}")

    if ch_reim.shape[1] < 2:
        raise ValueError("ch must have at least 2 numeric columns (Re, Im).")

    N = wgt_reim.shape[0]
    if ch_reim.shape[0] != N:
        raise ValueError(f"Row mismatch: ch has {ch_reim.shape[0]} rows, wgt has {N} rows.")

    if ch_mode == "row0":
        if not (0 <= ch_row < ch_reim.shape[0]):
            raise ValueError(f"ch_row out of range: {ch_row}")
        a = np.full((N, 1), ch_reim[ch_row, 0], dtype=float)
        b = np.full((N, 1), ch_reim[ch_row, 1], dtype=float)
    elif ch_mode == "per_row":
        a = ch_reim[:, 0:1]
        b = ch_reim[:, 1:2]
    else:
        raise ValueError("--ch_mode must be 'row0' or 'per_row'")

    re = wgt_reim[:, 0::2]
    im = wgt_reim[:, 1::2]

    out_re = a * re - b * im
    out_im = a * im + b * re

    out = np.empty_like(wgt_reim, dtype=float)
    out[:, 0::2] = out_re
    out[:, 1::2] = out_im
    return out


# -----------------------------
# Table logic: 48-row blocks
# -----------------------------
def split_48_blocks(mat: np.ndarray) -> List[np.ndarray]:
    n_rows, _ = mat.shape
    if n_rows % 48 != 0:
        raise ValueError(f"bm row count must be divisible by 48. got {n_rows}")
    n_blocks = n_rows // 48
    return [mat[i * 48 : (i + 1) * 48, :] for i in range(n_blocks)]


def sum_blocks(blocks: List[np.ndarray]) -> np.ndarray:
    return np.sum(np.stack(blocks, axis=0), axis=0)  # (48, C)


def compress_48_to_4(sum48: np.ndarray) -> np.ndarray:
    if sum48.shape[0] != 48:
        raise ValueError(f"Expected 48 rows, got {sum48.shape[0]}")
    return np.vstack([sum48[i::4, :].sum(axis=0) for i in range(4)])  # (4, C)


# -----------------------------
# Writing with blank line between tables
# -----------------------------
def write_csv_blocks(
    out_path: Path,
    blocks: List[np.ndarray],
    decimals: int = 13,
    blank_lines_between_blocks: int = 1,
) -> None:
    """
    Writes blocks with:
      - fixed decimals
      - trailing comma (to mimic original style)
      - N blank lines between 48-row tables
    """
    fmt = f"{{: .{decimals}f}}"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for bi, blk in enumerate(blocks):
            for r in range(blk.shape[0]):
                row = blk[r, :]
                line = ",  ".join(fmt.format(v) for v in row.tolist()) + ", \n"
                f.write(line)
            if bi != len(blocks) - 1:
                f.write("\n" * blank_lines_between_blocks)


def write_matrix_csv(out_path: Path, mat: np.ndarray, decimals: int = 13) -> None:
    fmt = f"{{: .{decimals}f}}"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for r in range(mat.shape[0]):
            line = ",  ".join(fmt.format(v) for v in mat[r, :].tolist()) + ", \n"
            f.write(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=str, required=True, help="Path to log_mult_ch.csv")
    ap.add_argument("--wgt", type=str, required=True, help="Path to log_mult_wgt.csv")
    ap.add_argument("--bm", type=str, default="log_mult_bm.csv",
                    help="If this bm exists, use it. If not, generate bm from ch+wgt and write to outdir.")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory")
    ap.add_argument("--bm_out", type=str, default="log_mult_bm.csv", help="bm output filename (in outdir)")
    ap.add_argument("--write_sum", action="store_true", help="Also write summed 48-row table")
    ap.add_argument("--sum_out", type=str, default="log_mult_bm_sum_48.csv", help="Summed 48-row output filename")
    ap.add_argument("--final_out", type=str, default="log_mult_final_4.csv", help="Final 4-row output filename")
    ap.add_argument("--decimals", type=int, default=13, help="Fixed decimals in outputs")
    ap.add_argument("--blank_lines", type=int, default=1, help="Blank lines between 48-row tables in bm")
    ap.add_argument("--ch_mode", type=str, default="row0", choices=["row0", "per_row"],
                    help="How to use ch: 'row0' (default) uses ch_row for all rows; 'per_row' uses each row's ch")
    ap.add_argument("--ch_row", type=int, default=0, help="Row index to use when ch_mode=row0")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ch_path = Path(args.ch)
    wgt_path = Path(args.wgt)
    bm_in_path = Path(args.bm)

    # Read ch + wgt
    ch_num, _ = extract_numeric(read_csv_str(ch_path))
    wgt_num, _ = extract_numeric(read_csv_str(wgt_path))

    # Keep only even number of columns for wgt (drop last if odd; but by spec it should be even)
    if wgt_num.shape[1] % 2 != 0:
        wgt_num = wgt_num[:, :-1]

    bm_out_path = outdir / args.bm_out

    # 1) Get bm numeric matrix
    if bm_in_path.exists():
        bm_num, _ = extract_numeric(read_csv_str(bm_in_path))
        if bm_num.shape[1] % 2 != 0:
            bm_num = bm_num[:, :-1]
        # also copy to outdir if needed
        if bm_in_path.resolve() != bm_out_path.resolve():
            # write a normalized copy with desired decimals + blank lines between tables
            blocks = split_48_blocks(bm_num)
            write_csv_blocks(bm_out_path, blocks, decimals=args.decimals, blank_lines_between_blocks=args.blank_lines)
        bm_matrix = bm_num
        src = f"used existing bm: {bm_in_path}"
    else:
        # generate from ch × wgt
        bm_matrix = complex_mul_rowwise(ch_num, wgt_num, ch_mode=args.ch_mode, ch_row=args.ch_row)
        blocks = split_48_blocks(bm_matrix)
        write_csv_blocks(bm_out_path, blocks, decimals=args.decimals, blank_lines_between_blocks=args.blank_lines)
        src = "generated bm from ch+wgt (no bm file found)"

    # 2) Sum across ALL 48-row tables (blocks)
    blocks = split_48_blocks(bm_matrix)
    sum48 = sum_blocks(blocks)

    if args.write_sum:
        write_matrix_csv(outdir / args.sum_out, sum48, decimals=args.decimals)

    # 3) Compress to 4 long rows
    final4 = compress_48_to_4(sum48)
    write_matrix_csv(outdir / args.final_out, final4, decimals=args.decimals)

    print(f"[OK] {src}")
    print(f"[OK] detected tables(blocks): {len(blocks)} (each 48 rows)")
    print(f"[OK] bm saved: {bm_out_path}")
    if args.write_sum:
        print(f"[OK] sum48 saved: {outdir / args.sum_out}")
    print(f"[OK] final4 saved: {outdir / args.final_out}")


if __name__ == "__main__":
    main()
