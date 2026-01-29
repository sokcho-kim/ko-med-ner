"""
Gazetteer 기반 자동 레이블링 스크립트

목적: 샘플링된 cg_parsed 데이터에 Gazetteer 매칭으로 엔티티 자동 태깅
출력: 레이블링된 JSONL (GLiNER 학습 형식 호환)

사용법:
    python label_with_gazetteer.py --input sampled.jsonl --output labeled.jsonl
"""

import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class GazetteerMatcher:
    """Gazetteer 기반 엔티티 매칭"""

    def __init__(self, gazetteer_dir: Path):
        self.gazetteer_dir = gazetteer_dir
        self.entries: Dict[str, Dict] = {}  # term -> {label, code, ...}
        self.terms_by_length: List[Tuple[str, str]] = []  # (term, label) sorted by length desc
        self.stats = defaultdict(int)

    def load_gazetteer(self, filename: str, label: str) -> int:
        """단일 Gazetteer 파일 로드"""
        path = self.gazetteer_dir / filename
        if not path.exists():
            print(f"  [SKIP] {filename} not found")
            return 0

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        entries = data.get('entries', {})
        count = 0

        for term, info in entries.items():
            # 너무 짧은 용어 제외 (노이즈 방지)
            if len(term) < 2:
                continue

            # 숫자만 있는 용어 제외
            if term.isdigit():
                continue

            # 중복 처리 (더 긴 label이 이미 있으면 유지)
            if term in self.entries:
                existing_label = self.entries[term]['label']
                # 같은 label이면 스킵
                if existing_label == label:
                    continue
                # 다른 label이면 우선순위: Disease > Drug > Procedure > Biomarker
                priority = {'Disease': 4, 'Drug': 3, 'Procedure': 2, 'Biomarker': 1}
                if priority.get(existing_label, 0) >= priority.get(label, 0):
                    continue

            self.entries[term] = {
                'label': label,
                'code': info.get('code', ''),
                'source': info.get('source', ''),
            }
            count += 1

        print(f"  [OK] {label}: {count} terms from {filename}")
        return count

    def load_all(self):
        """모든 Gazetteer 로드"""
        print("\n[Gazetteer 로드]")

        self.load_gazetteer('disease_gazetteer.json', 'Disease')
        self.load_gazetteer('drug_gazetteer.json', 'Drug')
        self.load_gazetteer('procedure_gazetteer.json', 'Procedure')
        self.load_gazetteer('biomarker_gazetteer.json', 'Biomarker')

        # 길이순 정렬 (긴 것부터 매칭 - greedy)
        self.terms_by_length = sorted(
            [(term, info['label']) for term, info in self.entries.items()],
            key=lambda x: len(x[0]),
            reverse=True
        )

        print(f"\n  총 {len(self.entries)} 용어 로드됨")
        print(f"  최장 용어: {len(self.terms_by_length[0][0])} chars")
        print(f"  최단 용어: {len(self.terms_by_length[-1][0])} chars")

    def find_matches(self, text: str) -> List[Dict]:
        """텍스트에서 모든 매칭 찾기 (중복 제거)"""
        matches = []
        used_positions: Set[int] = set()  # 이미 매칭된 위치

        # 길이순으로 매칭 (긴 것부터)
        for term, label in self.terms_by_length:
            start = 0
            while True:
                pos = text.find(term, start)
                if pos == -1:
                    break

                end = pos + len(term)

                # 이미 사용된 위치인지 확인
                overlap = False
                for i in range(pos, end):
                    if i in used_positions:
                        overlap = True
                        break

                if not overlap:
                    # 매칭 기록
                    matches.append({
                        'start': pos,
                        'end': end,
                        'text': term,
                        'label': label,
                        'code': self.entries[term]['code'],
                        'source': 'gazetteer'
                    })

                    # 위치 마킹
                    for i in range(pos, end):
                        used_positions.add(i)

                    self.stats[label] += 1

                start = pos + 1

        # 위치순 정렬
        matches.sort(key=lambda x: x['start'])
        return matches


def label_documents(
    input_path: Path,
    output_path: Path,
    matcher: GazetteerMatcher
) -> Dict:
    """문서 레이블링"""
    results = []
    stats = {
        'total': 0,
        'with_entities': 0,
        'entity_counts': defaultdict(int)
    }

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            stats['total'] += 1

            # 매칭 수행
            text = doc['text']
            matches = matcher.find_matches(text)

            # 결과 업데이트
            doc['entities'] = matches
            doc['labeled'] = True
            doc['labeled_by'] = 'gazetteer'
            doc['labeled_at'] = datetime.now().isoformat()

            if matches:
                stats['with_entities'] += 1
                for m in matches:
                    stats['entity_counts'][m['label']] += 1

            results.append(doc)

            # 진행 상황
            if stats['total'] % 100 == 0:
                print(f"  처리: {stats['total']}건...")

    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in results:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    return stats


def print_sample(output_path: Path, n: int = 3):
    """샘플 출력"""
    print(f"\n[샘플 미리보기]")

    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            doc = json.loads(line)
            if doc['entities']:
                print(f"\n--- Sample {i+1} ({doc['source']}) ---")
                print(f"ID: {doc['id']}")
                print(f"Text: {doc['text'][:100]}...")
                print(f"Entities ({len(doc['entities'])}):")
                for e in doc['entities'][:5]:
                    print(f"  - [{e['label']}] {e['text']} ({e['start']}-{e['end']})")
                if len(doc['entities']) > 5:
                    print(f"  ... and {len(doc['entities']) - 5} more")


def main():
    parser = argparse.ArgumentParser(description='Gazetteer 기반 자동 레이블링')
    parser.add_argument('--input', type=str, default=None, help='입력 JSONL 파일')
    parser.add_argument('--output', type=str, default=None, help='출력 JSONL 파일')
    args = parser.parse_args()

    # 경로 설정
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / 'data'
    gazetteer_dir = data_dir / 'gazetteer'
    silver_dir = data_dir / 'silver_set'

    if args.input:
        input_path = Path(args.input)
    else:
        # 가장 최근 샘플 파일 찾기
        samples = list(silver_dir.glob('cg_parsed_sampled_*.jsonl'))
        if not samples:
            print("ERROR: No sampled files found. Run sample_cg_parsed.py first.")
            return
        input_path = max(samples, key=lambda p: p.stat().st_mtime)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = silver_dir / f'{input_path.stem}_labeled.jsonl'

    print("=" * 60)
    print("Gazetteer 기반 자동 레이블링")
    print("=" * 60)
    print(f"\n입력: {input_path}")
    print(f"출력: {output_path}")

    # Gazetteer 로드
    matcher = GazetteerMatcher(gazetteer_dir)
    matcher.load_all()

    # 레이블링
    print("\n[레이블링 시작]")
    stats = label_documents(input_path, output_path, matcher)

    # 결과 출력
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"\n총 문서: {stats['total']}건")
    print(f"엔티티 있는 문서: {stats['with_entities']}건 ({stats['with_entities']/stats['total']*100:.1f}%)")
    print(f"\n엔티티 통계:")
    for label, count in sorted(stats['entity_counts'].items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}개")
    print(f"\n출력 파일: {output_path}")
    print(f"파일 크기: {output_path.stat().st_size / 1024:.1f} KB")

    # 샘플 출력
    print_sample(output_path)


if __name__ == '__main__':
    main()
