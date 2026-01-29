"""
GPT-5 기반 NER 레이블 검증 스크립트

태스크:
1. 오매칭 필터링 (GPT-5 nano) - False Positive 제거
2. 누락 엔티티 추가 (GPT-5 mini) - 커버리지 향상

기능:
- Rate Limit 자동 재시도 (exponential backoff)
- 중간 저장 (체크포인트, 기본 50건마다)
- 이어하기 (--resume)

사용법:
    # 오매칭 필터링만
    python verify_with_gpt5.py --task filter --input labeled.jsonl

    # 누락 엔티티 추가만
    python verify_with_gpt5.py --task add --input labeled.jsonl

    # 둘 다 (순차 실행)
    python verify_with_gpt5.py --task both --input labeled.jsonl

    # 파일럿 테스트 (10건만)
    python verify_with_gpt5.py --task both --limit 10

    # 병렬 처리 (3 workers, 권장)
    python verify_with_gpt5.py --task both --workers 3

    # 전체 1000건 실행 (권장 설정)
    python verify_with_gpt5.py --task both --workers 3

    # 중단된 작업 이어하기
    python verify_with_gpt5.py --task both --workers 3 --resume

    # 체크포인트 간격 조정 (기본 50건)
    python verify_with_gpt5.py --task both --checkpoint-interval 100
"""

import json
import argparse
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# .env 파일 로드 (dotenv 없이도 동작하도록 fallback 구현)
def _load_env_file(env_path: Path) -> dict:
    """수동으로 .env 파일 파싱 (dotenv fallback)"""
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # 따옴표 제거 및 CRLF 정리
                    value = value.strip().strip('"').strip("'").strip()
                    env_vars[key.strip()] = value
    return env_vars

# 프로젝트 루트 찾기 (__file__ 기준)
# scripts/data_prep -> scripts -> repo root (2 levels up)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
_env_path = _project_root / '.env'

try:
    from dotenv import load_dotenv
    load_dotenv(_env_path)
except ImportError:
    # dotenv 미설치 시 수동 로드
    _env_vars = _load_env_file(_env_path)
    for k, v in _env_vars.items():
        if k not in os.environ:  # 기존 환경변수 우선
            os.environ[k] = v

# OpenAI API (설치 필요: pip install openai)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("[WARNING] openai 패키지 미설치. pip install openai 실행 필요")


# ============================================================
# 프롬프트 정의
# ============================================================

FILTER_SYSTEM_PROMPT = """당신은 한국어 의료 텍스트의 개체명 인식(NER) 전문가입니다.
주어진 텍스트에서 자동으로 추출된 의료 엔티티들이 실제로 유효한지 검증해주세요.

엔티티 유형:
- Disease: 질병, 증상, 진단명 (예: 당뇨병, 폐암, 고혈압)
- Drug: 의약품, 성분명 (예: 메트포르민, 아스피린)
- Procedure: 수술, 검사, 처치 (예: CT, 내시경, 혈액검사)
- Biomarker: 바이오마커, 유전자 (예: HER2, EGFR, BRCA)

오매칭 판정 기준:
1. 부분 문자열: 단어의 일부만 추출된 경우 (예: "인터페론"에서 "론"만)
2. 동음이의어: 의료 용어가 아닌 일반 단어 (예: "주사"가 "injection"이 아닌 맥락)
3. 맥락 오류: 텍스트 맥락과 맞지 않는 레이블
4. 일반 단어: 의료 전문용어가 아닌 경우"""

FILTER_USER_PROMPT = """다음 텍스트에서 추출된 엔티티들을 검증해주세요.

[텍스트]
{text}

[추출된 엔티티]
{entities}

각 엔티티에 대해 JSON 형식으로 판정해주세요:
- "valid": true/false (유효 여부)
- "reason": 무효인 경우 이유 (부분문자열/동음이의어/맥락오류/일반단어)

응답 형식:
```json
{{
  "judgments": [
    {{"entity": "엔티티텍스트", "label": "레이블", "valid": true/false, "reason": "이유 또는 null"}}
  ]
}}
```"""

ADD_SYSTEM_PROMPT = """당신은 한국어 의료 텍스트의 개체명 인식(NER) 전문가입니다.
주어진 텍스트에서 누락된 의료 엔티티를 찾아주세요.

추출 대상 엔티티:
- Disease: 질병, 증상, 진단명, 증후군
- Drug: 의약품명, 성분명, 제제명
- Procedure: 수술, 검사, 처치, 시술
- Biomarker: 바이오마커, 유전자, 수용체

주의사항:
1. 이미 추출된 엔티티는 제외
2. 명확한 의료 용어만 추출
3. 일반 단어나 모호한 표현은 제외
4. 엔티티의 정확한 위치(시작, 끝)를 표시"""

ADD_USER_PROMPT = """다음 텍스트에서 누락된 의료 엔티티를 찾아주세요.

[텍스트]
{text}

[이미 추출된 엔티티]
{entities}

누락된 엔티티를 JSON 형식으로 응답해주세요:
- "text": 엔티티 텍스트
- "label": Disease/Drug/Procedure/Biomarker
- "start": 시작 위치 (0-indexed)
- "end": 끝 위치

응답 형식:
```json
{{
  "missing_entities": [
    {{"text": "엔티티", "label": "레이블", "start": 0, "end": 5}}
  ]
}}
```

누락된 엔티티가 없으면 빈 배열을 반환하세요."""


# ============================================================
# GPT-5 API 호출
# ============================================================

@dataclass
class GPT5Config:
    """GPT-5 모델 설정"""
    nano: str = "gpt-5-nano"      # 필터링용 (저비용)
    mini: str = "gpt-5-mini"      # 추가용 (중비용)
    standard: str = "gpt-5"       # 고품질 검증용


class GPT5Verifier:
    """GPT-5 기반 검증기 (Thread-safe)"""

    def __init__(self, api_key: Optional[str] = None):
        if not HAS_OPENAI:
            raise ImportError("openai 패키지가 필요합니다. pip install openai")

        import httpx
        # 타임아웃: connect 10초, read 120초, write 30초
        timeout = httpx.Timeout(120.0, connect=10.0, read=120.0, write=30.0)
        # API 키에서 CRLF 줄바꿈 문자 제거 (.env 파일이 Windows 형식일 수 있음)
        resolved_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self.client = OpenAI(
            api_key=resolved_key,
            timeout=timeout,
            max_retries=0  # 자체 재시도 비활성화 (우리 코드에서 처리)
        )
        self.config = GPT5Config()
        self.stats = {
            'filter': {'calls': 0, 'input_tokens': 0, 'output_tokens': 0},
            'add': {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}
        }
        self._lock = threading.Lock()  # Thread-safe stats update

    def _call_api(self, model: str, system: str, user: str, max_retries: int = 10) -> Dict:
        """API 호출 (강화된 재시도 로직 포함)"""
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "response_format": {"type": "json_object"}
        }

        if not model.startswith("gpt-5"):
            params["temperature"] = 0.1

        # 재시도 횟수 5 -> 10회로 증가
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                usage = response.usage
                # 성공 시 1초 대기 (안정성 확보)
                time.sleep(1.0)
                return {
                    'content': response.choices[0].message.content,
                    'input_tokens': usage.prompt_tokens,
                    'output_tokens': usage.completion_tokens
                }
            except Exception as e:
                error_msg = str(e).lower()
                # 연결 에러나 서버 에러(5xx)인지 확인
                is_connection_issue = any(x in error_msg for x in ['connection', 'connect', 'timeout', '500', '503'])

                if is_connection_issue or 'rate' in error_msg or '429' in error_msg:
                    # Connection 에러면 30초부터 시작해서 기하급수적으로 대기 (30, 60, 120...)
                    # Rate Limit이면 2초부터 시작
                    base_wait = 30 if is_connection_issue else 2
                    wait_time = base_wait * (2 ** attempt)

                    # 최대 대기 시간은 3분으로 제한
                    wait_time = min(wait_time, 180)

                    print(f"    [RETRY] {type(e).__name__} ({str(e)[:30]}...) -> {wait_time}초 대기 후 재시도 ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    # 복구 불가능한 에러(JSON 파싱 오류 등)는 즉시 중단
                    raise

        raise Exception(f"Max retries ({max_retries}) exceeded")

    def filter_entities(self, text: str, entities: List[Dict]) -> List[Dict]:
        """오매칭 필터링 (GPT-5 nano)"""
        if not entities:
            return []

        # 프롬프트 구성
        entity_str = "\n".join([
            f"- [{e['label']}] \"{e['text']}\" (위치: {e['start']}-{e['end']})"
            for e in entities
        ])

        user_prompt = FILTER_USER_PROMPT.format(
            text=text[:2000],  # 토큰 제한
            entities=entity_str
        )

        # API 호출
        result = self._call_api(self.config.nano, FILTER_SYSTEM_PROMPT, user_prompt)
        with self._lock:
            self.stats['filter']['calls'] += 1
            self.stats['filter']['input_tokens'] += result['input_tokens']
            self.stats['filter']['output_tokens'] += result['output_tokens']

        # 결과 파싱
        try:
            data = json.loads(result['content'])
            judgments = {j['entity']: j for j in data.get('judgments', [])}

            # 유효한 엔티티만 반환
            valid_entities = []
            for e in entities:
                j = judgments.get(e['text'], {'valid': True})
                if j.get('valid', True):
                    valid_entities.append(e)
                else:
                    e['filtered'] = True
                    e['filter_reason'] = j.get('reason', 'unknown')

            return valid_entities
        except json.JSONDecodeError:
            return entities  # 파싱 실패 시 원본 반환

    def add_missing_entities(self, text: str, entities: List[Dict]) -> List[Dict]:
        """누락 엔티티 추가 (GPT-5 mini)"""
        # 프롬프트 구성
        entity_str = "\n".join([
            f"- [{e['label']}] \"{e['text']}\" (위치: {e['start']}-{e['end']})"
            for e in entities
        ]) if entities else "(없음)"

        user_prompt = ADD_USER_PROMPT.format(
            text=text[:2000],
            entities=entity_str
        )

        # API 호출
        result = self._call_api(self.config.mini, ADD_SYSTEM_PROMPT, user_prompt)
        with self._lock:
            self.stats['add']['calls'] += 1
            self.stats['add']['input_tokens'] += result['input_tokens']
            self.stats['add']['output_tokens'] += result['output_tokens']

        # 결과 파싱
        try:
            data = json.loads(result['content'])
            new_entities = data.get('missing_entities', [])

            # 새 엔티티에 메타데이터 추가
            for e in new_entities:
                e['source'] = 'gpt5-mini'
                e['code'] = ''  # 코드는 나중에 매칭

            return entities + new_entities
        except json.JSONDecodeError:
            return entities

    def get_cost_estimate(self) -> Dict:
        """비용 추정"""
        # GPT-5 가격 (2026년 기준)
        prices = {
            'nano': {'input': 0.05 / 1_000_000, 'output': 0.40 / 1_000_000},
            'mini': {'input': 0.25 / 1_000_000, 'output': 2.00 / 1_000_000},
        }

        filter_cost = (
            self.stats['filter']['input_tokens'] * prices['nano']['input'] +
            self.stats['filter']['output_tokens'] * prices['nano']['output']
        )

        add_cost = (
            self.stats['add']['input_tokens'] * prices['mini']['input'] +
            self.stats['add']['output_tokens'] * prices['mini']['output']
        )

        return {
            'filter': {'cost': filter_cost, **self.stats['filter']},
            'add': {'cost': add_cost, **self.stats['add']},
            'total': filter_cost + add_cost
        }


# ============================================================
# 메인 프로세스
# ============================================================

def process_single_document(
    doc: Dict,
    verifier: GPT5Verifier,
    task: str
) -> tuple:
    """단일 문서 처리 (병렬 처리용)"""
    original_count = len(doc.get('entities', []))
    filtered = 0
    added = 0

    try:
        # 1. 오매칭 필터링
        if task in ['filter', 'both']:
            doc['entities'] = verifier.filter_entities(
                doc['text'],
                doc.get('entities', [])
            )
            filtered = original_count - len(doc['entities'])

        # 2. 누락 엔티티 추가
        if task in ['add', 'both']:
            before_add = len(doc['entities'])
            doc['entities'] = verifier.add_missing_entities(
                doc['text'],
                doc.get('entities', [])
            )
            added = len(doc['entities']) - before_add

        # 메타데이터 업데이트
        doc['verified'] = True
        doc['verified_by'] = f'gpt5-{task}'
        doc['verified_at'] = datetime.now().isoformat()

        return doc, filtered, added, None
    except Exception as e:
        return doc, 0, 0, str(e)


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """체크포인트 로드"""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'processed_ids': [], 'results': {}, 'stats': {
        'total': 0, 'filtered_entities': 0, 'added_entities': 0, 'errors': 0
    }}


def save_checkpoint(checkpoint_path: Path, checkpoint: Dict):
    """체크포인트 저장"""
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def process_documents(
    input_path: Path,
    output_path: Path,
    verifier: GPT5Verifier,
    task: str = 'both',
    limit: Optional[int] = None,
    workers: int = 1,
    resume: bool = False,
    checkpoint_interval: int = 50
) -> Dict:
    """문서 처리 (배치 단위 병렬 처리로 안정성 강화)"""

    checkpoint_path = output_path.parent / f'.checkpoint_{output_path.stem}.json'

    # 백업 로직
    if not resume and output_path.exists():
        backup_name = f"{output_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{output_path.suffix}"
        import shutil
        shutil.copy2(output_path, output_path.parent / backup_name)
        print(f"  [BACKUP] 기존 파일 백업됨: {backup_name}")

    # 체크포인트 로드
    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        print(f"  [RESUME] 체크포인트 발견: {len(checkpoint['processed_ids'])}건 처리됨")
    else:
        checkpoint = load_checkpoint(Path('/nonexistent'))

    stats = checkpoint['stats']

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if limit:
        lines = lines[:limit]

    docs = [json.loads(line) for line in lines]
    total = len(docs)

    processed_ids = set(checkpoint['processed_ids'])
    docs_to_process = [(i, doc) for i, doc in enumerate(docs) if i not in processed_ids]

    if not docs_to_process:
        print("  모든 문서가 이미 처리되었습니다.")
        return stats

    print(f"  처리 대상: {len(docs_to_process)}건 (스킵: {len(processed_ids)}건)")

    results_dict = checkpoint['results'].copy()
    save_lock = threading.Lock()
    processed_count = len(processed_ids)

    def save_result(idx, doc, filtered, added, error):
        nonlocal processed_count
        with save_lock:
            if error:
                stats['errors'] += 1
                print(f"  [SKIP] Doc {idx} - 에러: {error}")
            else:
                results_dict[str(idx)] = doc
                checkpoint['processed_ids'].append(idx)
                checkpoint['results'] = results_dict
                stats['total'] += 1
                stats['filtered_entities'] += filtered
                stats['added_entities'] += added

            processed_count += 1
            # 매 건마다 체크포인트 저장 (안정성 우선)
            try:
                save_checkpoint(checkpoint_path, checkpoint)
            except PermissionError as e:
                print(f"  [ERROR] 체크포인트 저장 실패: {e}")

            if processed_count % checkpoint_interval == 0:
                print(f"  [CHECKPOINT] {len(checkpoint['processed_ids'])}/{total}건 저장됨")

    start_time = datetime.now()

    # --- [핵심 수정: 데이터를 청크(Chunk)로 나누어 처리] ---
    # 병렬이든 순차든 50개씩 끊어서 처리하고 휴식 시간을 가짐
    chunk_size = 50
    rest_seconds = 20  # 청크 간 휴식 시간

    # 전체 작업을 chunk_size 단위로 나눔
    chunks = [docs_to_process[i:i + chunk_size] for i in range(0, len(docs_to_process), chunk_size)]

    if HAS_TQDM:
        pbar = tqdm(total=len(docs_to_process), desc="검증 진행", unit="건", initial=0)

    for chunk_idx, chunk in enumerate(chunks):
        # 1. 청크 처리 (병렬 또는 순차)
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_idx = {
                    executor.submit(process_single_document, doc, verifier, task): idx
                    for idx, doc in chunk
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        r_doc, filt, add, err = future.result()
                        save_result(idx, r_doc, filt, add, err)
                    except Exception as e:
                        save_result(idx, docs[idx], 0, 0, str(e))

                    if HAS_TQDM: pbar.update(1)
        else:
            # 순차 처리
            for idx, doc in chunk:
                r_doc, filt, add, err = process_single_document(doc, verifier, task)
                save_result(idx, r_doc, filt, add, err)
                if HAS_TQDM: pbar.update(1)

        # 2. 청크 하나가 끝나면 휴식 (마지막 청크 제외)
        if chunk_idx < len(chunks) - 1:
            msg = f"  [REST] {rest_seconds}초 대기 (API 연결 안정화)..."
            if HAS_TQDM:
                pbar.write(msg)
            else:
                print(msg)
            time.sleep(rest_seconds)

    if HAS_TQDM:
        pbar.close()

    # 최종 저장
    save_checkpoint(checkpoint_path, checkpoint)

    # 결과 파일 쓰기
    final_results = [results_dict.get(str(i), docs[i]) for i in range(total)]
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in final_results:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    if checkpoint_path.exists() and stats['errors'] == 0:
        checkpoint_path.unlink()
        print("  [CLEANUP] 체크포인트 삭제됨")

    return stats


def dry_run(input_path: Path, limit: int = 5):
    """API 호출 없이 프롬프트만 출력 (테스트용)"""
    print("\n" + "=" * 60)
    print("[DRY RUN] API 호출 없이 프롬프트 미리보기")
    print("=" * 60)

    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            doc = json.loads(line)
            entities = doc.get('entities', [])

            if not entities:
                continue

            print(f"\n--- Document {i + 1} ---")
            print(f"Text: {doc['text'][:200]}...")

            # 필터링 프롬프트
            entity_str = "\n".join([
                f"- [{e['label']}] \"{e['text']}\""
                for e in entities[:5]
            ])

            print(f"\n[Filter Prompt]")
            print(f"엔티티 {len(entities)}개 검증 요청:")
            print(entity_str)
            if len(entities) > 5:
                print(f"  ... 외 {len(entities) - 5}개")


def main():
    parser = argparse.ArgumentParser(description='GPT-5 기반 NER 레이블 검증')
    parser.add_argument('--task', choices=['filter', 'add', 'both'], default='both',
                       help='실행할 태스크')
    parser.add_argument('--input', type=str, default=None, help='입력 JSONL 파일')
    parser.add_argument('--output', type=str, default=None, help='출력 JSONL 파일')
    parser.add_argument('--limit', type=int, default=None, help='처리할 문서 수 제한')
    parser.add_argument('--workers', type=int, default=3, help='병렬 처리 worker 수 (기본: 3, 권장: 3-5)')
    parser.add_argument('--resume', action='store_true', help='중단된 작업 이어하기')
    parser.add_argument('--checkpoint-interval', type=int, default=50, help='체크포인트 저장 간격 (기본: 50)')
    parser.add_argument('--dry-run', action='store_true', help='API 호출 없이 테스트')
    args = parser.parse_args()

    # 경로 설정
    base_dir = Path(__file__).parent.parent.parent
    silver_dir = base_dir / 'data' / 'silver_set'

    if args.input:
        input_path = Path(args.input)
    else:
        # 가장 최근 labeled 파일 찾기
        labeled = list(silver_dir.glob('*_labeled.jsonl'))
        if not labeled:
            print("ERROR: No labeled files found. Run label_with_gazetteer.py first.")
            return
        input_path = max(labeled, key=lambda p: p.stat().st_mtime)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = silver_dir / f'{input_path.stem}_verified.jsonl'

    print("=" * 60)
    print("GPT-5 기반 NER 레이블 검증")
    print("=" * 60)
    print(f"\n태스크: {args.task}")
    print(f"입력: {input_path}")
    print(f"출력: {output_path}")
    print(f"Workers: {args.workers}")
    print(f"Resume: {args.resume}")
    print(f"Checkpoint interval: {args.checkpoint_interval}건")

    # Dry run
    if args.dry_run:
        dry_run(input_path, args.limit or 5)
        return

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[ERROR] OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("export OPENAI_API_KEY='your-api-key' 실행 후 다시 시도하세요.")
        print("\n또는 --dry-run 옵션으로 프롬프트만 확인할 수 있습니다.")
        return

    # 검증 실행
    verifier = GPT5Verifier()

    print(f"\n[검증 시작] (limit: {args.limit or 'all'}, workers: {args.workers})")
    start_time = datetime.now()
    stats = process_documents(
        input_path, output_path, verifier, args.task, args.limit, args.workers,
        resume=args.resume, checkpoint_interval=args.checkpoint_interval
    )
    elapsed = (datetime.now() - start_time).total_seconds()

    # 비용 계산
    cost = verifier.get_cost_estimate()

    # 결과 출력
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"\n처리 문서: {stats['total']}건")
    print(f"필터링된 엔티티: {stats['filtered_entities']}개")
    print(f"추가된 엔티티: {stats['added_entities']}개")
    if stats.get('errors', 0) > 0:
        print(f"오류: {stats['errors']}건")

    print(f"\n[소요 시간]")
    print(f"  총 시간: {elapsed/60:.1f}분 ({elapsed:.0f}초)")
    print(f"  처리 속도: {stats['total']/elapsed:.2f}건/초")

    print(f"\n[API 사용량]")
    print(f"  Filter (nano): {cost['filter']['calls']} calls, "
          f"{cost['filter']['input_tokens']:,} in / {cost['filter']['output_tokens']:,} out, "
          f"${cost['filter']['cost']:.4f}")
    print(f"  Add (mini): {cost['add']['calls']} calls, "
          f"{cost['add']['input_tokens']:,} in / {cost['add']['output_tokens']:,} out, "
          f"${cost['add']['cost']:.4f}")
    print(f"  총 비용: ${cost['total']:.4f}")

    print(f"\n출력 파일: {output_path}")


if __name__ == '__main__':
    main()
