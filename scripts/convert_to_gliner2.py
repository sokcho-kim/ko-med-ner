"""
Silver-set 데이터를 GLiNER2 학습 형식으로 변환

입력: cg_parsed_sampled_1000_labeled_verified.jsonl
출력: GLiNER2 InputExample 형식의 JSONL

사용법:
    # 검증된 데이터만 (기본)
    python convert_to_gliner2.py

    # 전체 데이터 (미검증 포함)
    python convert_to_gliner2.py --all

    # Train/Test 분할 비율 조정
    python convert_to_gliner2.py --test-ratio 0.1

    # 출력 경로 지정
    python convert_to_gliner2.py --output ./my_output
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ============================================
# 설정
# ============================================

# 엔티티 라벨 설명 (GLiNER2 학습에 사용)
ENTITY_DESCRIPTIONS = {
    "Disease": "질병, 증상, 의학적 상태 (예: 당뇨병, 고혈압, 폐렴, 암)",
    "Drug": "약물, 의약품, 치료제 (예: 인슐린, 아스피린, 항생제)",
    "Procedure": "의료 시술, 수술, 검사 (예: 내시경, MRI, 수술)",
    "Biomarker": "바이오마커, 검사 수치, 생체 지표 (예: 혈당, 콜레스테롤, 종양표지자)"
}

# 기본 경로
DEFAULT_INPUT = Path(__file__).parent.parent / "data/silver_set/cg_parsed_sampled_1000_labeled_verified.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data/gliner2_train"


def load_silver_data(input_path: Path, verified_only: bool = True) -> list:
    """Silver-set 데이터 로드"""
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            if verified_only and not doc.get('verified'):
                continue
            data.append(doc)
    return data


def convert_to_gliner2_format(doc: dict) -> dict:
    """
    Silver-set 문서를 GLiNER2 형식으로 변환

    Silver-set 형식:
    {
        "text": "...",
        "entities": [
            {"text": "당뇨병", "label": "Disease", "start": 0, "end": 3},
            ...
        ]
    }

    GLiNER2 형식:
    {
        "text": "...",
        "entities": {
            "Disease": ["당뇨병", ...],
            "Drug": [...],
            ...
        },
        "entity_descriptions": {...}
    }
    """
    text = doc['text']

    # 라벨별로 엔티티 그룹화
    entities_by_label = defaultdict(list)
    for ent in doc.get('entities', []):
        label = ent['label']
        mention = ent['text']
        # 중복 제거
        if mention not in entities_by_label[label]:
            entities_by_label[label].append(mention)

    # GLiNER2 형식으로 변환
    gliner2_doc = {
        "text": text,
        "entities": dict(entities_by_label),
        "entity_descriptions": ENTITY_DESCRIPTIONS
    }

    # 메타데이터 (옵션)
    if 'id' in doc:
        gliner2_doc['id'] = doc['id']

    return gliner2_doc


def split_train_test(data: list, test_ratio: float = 0.1, seed: int = 42) -> tuple:
    """Train/Test 분할"""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    test_size = int(len(shuffled) * test_ratio)
    test_data = shuffled[:test_size]
    train_data = shuffled[test_size:]

    return train_data, test_data


def save_jsonl(data: list, output_path: Path):
    """JSONL 형식으로 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')


def print_stats(data: list, name: str):
    """데이터 통계 출력"""
    total_entities = 0
    label_counts = defaultdict(int)

    for doc in data:
        for label, mentions in doc['entities'].items():
            label_counts[label] += len(mentions)
            total_entities += len(mentions)

    print(f"\n{name} 통계:")
    print(f"  문서 수: {len(data)}")
    print(f"  총 엔티티: {total_entities}")
    for label in sorted(label_counts.keys()):
        print(f"    - {label}: {label_counts[label]}")


def main():
    parser = argparse.ArgumentParser(description='Silver-set → GLiNER2 형식 변환')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help='입력 파일 경로')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='출력 디렉토리')
    parser.add_argument('--all', action='store_true',
                        help='미검증 데이터 포함 (기본: 검증된 것만)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='테스트셋 비율 (기본: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드')
    args = parser.parse_args()

    print("=" * 60)
    print("Silver-set → GLiNER2 형식 변환")
    print("=" * 60)

    # 1. 데이터 로드
    verified_only = not args.all
    print(f"\n[1/4] 데이터 로드...")
    print(f"  입력: {args.input}")
    print(f"  모드: {'검증된 데이터만' if verified_only else '전체 데이터'}")

    raw_data = load_silver_data(args.input, verified_only=verified_only)
    print(f"  로드됨: {len(raw_data)}건")

    if len(raw_data) == 0:
        print("\n[ERROR] 데이터가 없습니다!")
        return

    # 2. GLiNER2 형식으로 변환
    print(f"\n[2/4] GLiNER2 형식 변환...")
    converted_data = [convert_to_gliner2_format(doc) for doc in raw_data]

    # 엔티티 없는 문서 제외
    valid_data = [doc for doc in converted_data if doc['entities']]
    excluded = len(converted_data) - len(valid_data)
    if excluded > 0:
        print(f"  엔티티 없는 문서 제외: {excluded}건")
    print(f"  유효 문서: {len(valid_data)}건")

    # 3. Train/Test 분할
    print(f"\n[3/4] Train/Test 분할 (test_ratio={args.test_ratio})...")
    train_data, test_data = split_train_test(valid_data, args.test_ratio, args.seed)
    print(f"  Train: {len(train_data)}건")
    print(f"  Test: {len(test_data)}건")

    # 4. 저장
    print(f"\n[4/4] 저장...")
    train_path = args.output / "train.jsonl"
    test_path = args.output / "test.jsonl"

    save_jsonl(train_data, train_path)
    save_jsonl(test_data, test_path)

    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    # 통계 출력
    print_stats(train_data, "Train")
    print_stats(test_data, "Test")

    # 메타 정보 저장
    meta = {
        "created_at": datetime.now().isoformat(),
        "source": str(args.input),
        "verified_only": verified_only,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "train_count": len(train_data),
        "test_count": len(test_data),
        "entity_descriptions": ENTITY_DESCRIPTIONS
    }
    meta_path = args.output / "meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n메타 정보: {meta_path}")

    print("\n" + "=" * 60)
    print("변환 완료!")
    print("=" * 60)

    # 샘플 출력
    print("\n샘플 (train.jsonl 첫 번째):")
    sample = train_data[0]
    print(f"  text: {sample['text'][:60]}...")
    print(f"  entities: {sample['entities']}")


if __name__ == "__main__":
    main()
