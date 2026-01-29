"""
cg_parsed 데이터 샘플링 스크립트

목적: GLiNER2 학습/평가용 NER 데이터셋 구축을 위한 샘플 추출
출력: JSONL 형식 (Gazetteer 매칭 및 LLM 레이블링 입력용)

사용법:
    python sample_cg_parsed.py --gosi 1000 --case 1000 --output sampled_data.jsonl
"""

import pandas as pd
import json
import argparse
import os
import random
from pathlib import Path
from datetime import datetime


def safe_str(series: pd.Series) -> pd.Series:
    """NaN을 빈 문자열로 변환"""
    return series.fillna('').astype(str)


def load_gosi_data(path: str) -> pd.DataFrame:
    """고시 데이터 로드"""
    df = pd.read_excel(path)

    # 텍스트 필드 선택 (updated_content가 메인, revision_reason 보조)
    content = safe_str(df['updated_content'])
    reason = safe_str(df['revision_reason'])
    df['text'] = (content + '\n' + reason).str.strip()

    # 메타데이터 추가
    df['source'] = 'gosi'
    doc_id = safe_str(df['notification_number'])
    df['doc_id'] = doc_id.where(doc_id != '', df.index.astype(str))
    df['doc_title'] = safe_str(df['notification_title'])
    df['doc_date'] = df['publication_date']

    return df[['doc_id', 'doc_title', 'doc_date', 'text', 'source', 'url']]


def load_case_data(path: str) -> pd.DataFrame:
    """사례 데이터 로드"""
    df = pd.read_excel(path)

    # 텍스트 필드 결합 (case_content + decision_reason)
    content = safe_str(df['case_content'])
    reason = safe_str(df['decision_reason'])
    df['text'] = (content + '\n' + reason).str.strip()

    # 메타데이터 추가
    df['source'] = 'case'
    df['doc_id'] = df.index.astype(str)
    df['doc_title'] = safe_str(df['title'])
    df['doc_date'] = df['publication_date']

    return df[['doc_id', 'doc_title', 'doc_date', 'text', 'source', 'url']]


def filter_by_length(df: pd.DataFrame, min_len: int = 50, max_len: int = 5000) -> pd.DataFrame:
    """텍스트 길이로 필터링"""
    df = df[df['text'].str.len() >= min_len]
    df = df[df['text'].str.len() <= max_len]
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """텍스트 길이 기준 층화 샘플링"""
    random.seed(seed)

    # 길이 구간별 분류
    df = df.copy()
    df['len_bin'] = pd.cut(
        df['text'].str.len(),
        bins=[0, 100, 300, 1000, 5000],
        labels=['short', 'medium', 'long', 'very_long']
    )

    # 구간별 비율 유지하며 샘플링
    sampled = df.groupby('len_bin', group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(1, int(n * len(x) / len(df)))), random_state=seed)
    )

    # 부족분 랜덤 추가
    if len(sampled) < n:
        remaining = df[~df.index.isin(sampled.index)]
        additional = remaining.sample(min(n - len(sampled), len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, additional])

    return sampled.drop(columns=['len_bin']).head(n)


def to_jsonl(df: pd.DataFrame, output_path: str):
    """JSONL 형식으로 저장"""
    records = []
    for idx, row in df.iterrows():
        record = {
            'id': f"{row['source']}_{row['doc_id']}",
            'source': row['source'],
            'doc_id': row['doc_id'],
            'doc_title': row['doc_title'],
            'doc_date': str(row['doc_date']) if pd.notna(row['doc_date']) else None,
            'url': row['url'] if pd.notna(row['url']) else None,
            'text': row['text'],
            'text_length': len(row['text']),
            # NER 레이블링용 빈 필드
            'entities': [],
            'labeled': False,
            'labeled_by': None,
            'labeled_at': None
        }
        records.append(record)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    return records


def print_stats(df: pd.DataFrame, name: str):
    """통계 출력"""
    print(f"\n=== {name} ===")
    print(f"  샘플 수: {len(df)}")
    print(f"  텍스트 길이:")
    print(f"    - 평균: {df['text'].str.len().mean():.0f}")
    print(f"    - 중앙값: {df['text'].str.len().median():.0f}")
    print(f"    - 최소: {df['text'].str.len().min():.0f}")
    print(f"    - 최대: {df['text'].str.len().max():.0f}")


def main():
    parser = argparse.ArgumentParser(description='cg_parsed 데이터 샘플링')
    parser.add_argument('--gosi', type=int, default=1000, help='고시 샘플 수')
    parser.add_argument('--case', type=int, default=1000, help='사례 샘플 수')
    parser.add_argument('--min-len', type=int, default=50, help='최소 텍스트 길이')
    parser.add_argument('--max-len', type=int, default=5000, help='최대 텍스트 길이')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--output', type=str, default=None, help='출력 파일 경로')
    args = parser.parse_args()

    # 경로 설정
    repo_root = Path(__file__).parent.parent.parent
    external_root = Path(os.environ.get("SCRAPE_HUB_ROOT", "C:/Jimin/scrape-hub"))
    data_dir = external_root / 'data' / 'cg_parsed'
    output_dir = repo_root / 'data' / 'silver_set'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / f'cg_parsed_sampled_{args.gosi + args.case}.jsonl'

    print("=" * 60)
    print("cg_parsed 샘플링")
    print("=" * 60)

    # 데이터 로드
    print("\n[1/4] 데이터 로드 중...")
    gosi_path = data_dir / '고시_20251101.xlsx'
    case_path = data_dir / '사례_20251101.xlsx'

    df_gosi = load_gosi_data(str(gosi_path))
    df_case = load_case_data(str(case_path))

    print(f"  고시: {len(df_gosi)}건")
    print(f"  사례: {len(df_case)}건")

    # 필터링
    print("\n[2/4] 길이 필터링...")
    df_gosi = filter_by_length(df_gosi, args.min_len, args.max_len)
    df_case = filter_by_length(df_case, args.min_len, args.max_len)

    print(f"  고시 (필터 후): {len(df_gosi)}건")
    print(f"  사례 (필터 후): {len(df_case)}건")

    # 샘플링
    print("\n[3/4] 층화 샘플링...")
    sampled_gosi = stratified_sample(df_gosi, args.gosi, args.seed)
    sampled_case = stratified_sample(df_case, args.case, args.seed)

    print_stats(sampled_gosi, "고시 샘플")
    print_stats(sampled_case, "사례 샘플")

    # 병합
    sampled_all = pd.concat([sampled_gosi, sampled_case], ignore_index=True)

    # 저장
    print(f"\n[4/4] 저장 중... → {output_path}")
    records = to_jsonl(sampled_all, str(output_path))

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"\n총 샘플: {len(records)}건")
    print(f"  - 고시: {len(sampled_gosi)}건")
    print(f"  - 사례: {len(sampled_case)}건")
    print(f"\n출력 파일: {output_path}")
    print(f"파일 크기: {output_path.stat().st_size / 1024:.1f} KB")

    # 샘플 미리보기
    print("\n[샘플 미리보기]")
    for i, record in enumerate(records[:2]):
        print(f"\n--- Sample {i+1} ({record['source']}) ---")
        print(f"ID: {record['id']}")
        print(f"Title: {record['doc_title'][:50]}..." if record['doc_title'] and len(record['doc_title']) > 50 else f"Title: {record['doc_title']}")
        print(f"Text ({record['text_length']} chars): {record['text'][:200]}...")


if __name__ == '__main__':
    main()
