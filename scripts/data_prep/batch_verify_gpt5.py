"""
GPT-5 Batch API 기반 NER 검증 스크립트

OpenAI Batch API를 사용하여 1000건을 한 번에 처리
- 50% 비용 절감
- 타임아웃 없음
- 24시간 내 자동 완료

사용법:
    # 배치 요청 생성 및 제출
    python batch_verify_gpt5.py submit --input labeled.jsonl

    # 배치 상태 확인
    python batch_verify_gpt5.py status --batch-id batch_xxx

    # 결과 다운로드
    python batch_verify_gpt5.py download --batch-id batch_xxx --output verified.jsonl
"""

import json
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# .env 로드
def _load_env_file(env_path: Path) -> dict:
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'").strip()
                    env_vars[key.strip()] = value
    return env_vars

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
_env_path = _project_root / '.env'

try:
    from dotenv import load_dotenv
    load_dotenv(_env_path)
except ImportError:
    _env_vars = _load_env_file(_env_path)
    for k, v in _env_vars.items():
        if k not in os.environ:
            os.environ[k] = v

from openai import OpenAI

# 프롬프트 (verify_with_gpt5.py와 동일)
FILTER_SYSTEM_PROMPT = """당신은 한국어 의료 텍스트의 개체명 인식(NER) 전문가입니다.

주어진 텍스트에서 추출된 엔티티 목록을 검토하고, 잘못 추출된 엔티티(False Positive)를 식별하세요.

## 레이블 정의
- Disease: 질병, 증상, 진단명 (예: 당뇨병, 고혈압, 두통)
- Drug: 약물명, 성분명, 제품명 (예: 아스피린, 메트포르민)
- Procedure: 의료 시술, 검사, 수술명 (예: CT촬영, 혈액검사, 위내시경)
- Biomarker: 바이오마커, 유전자, 수용체 (예: HER2, EGFR, BRCA)

## 판단 기준
엔티티가 False Positive인 경우:
1. 해당 레이블에 맞지 않는 경우 (예: 약물이 아닌데 Drug으로 태깅)
2. 의료 용어가 아닌 일반 명사인 경우
3. 부분만 추출되어 의미가 불완전한 경우
4. 오타나 인코딩 오류로 의미 파악이 불가능한 경우

## 응답 형식
JSON 배열로 제거할 엔티티의 인덱스를 반환하세요.
예: [0, 2, 5] (0번, 2번, 5번 엔티티 제거)
제거할 것이 없으면: []"""

ADD_SYSTEM_PROMPT = """당신은 한국어 의료 텍스트의 개체명 인식(NER) 전문가입니다.

주어진 텍스트에서 누락된 의료 엔티티를 찾아 추가하세요.

## 레이블 정의
- Disease: 질병, 증상, 진단명
- Drug: 약물명, 성분명, 제품명
- Procedure: 의료 시술, 검사, 수술명
- Biomarker: 바이오마커, 유전자, 수용체

## 주의사항
1. 이미 추출된 엔티티는 다시 추가하지 마세요
2. 명확한 의료 용어만 추가하세요
3. 일반 명사나 애매한 표현은 제외하세요

## 응답 형식
JSON 배열로 추가할 엔티티를 반환하세요.
예: [{"text": "고혈압", "label": "Disease", "start": 10, "end": 13}]
추가할 것이 없으면: []"""


def create_batch_requests(input_file: Path, output_file: Path, task: str = "both"):
    """JSONL 파일에서 배치 요청 생성"""

    with open(input_file, 'r', encoding='utf-8') as f:
        docs = [json.loads(line) for line in f]

    requests = []

    for idx, doc in enumerate(docs):
        text = doc.get('text', '')
        entities = doc.get('entities', [])

        if task in ['filter', 'both']:
            # 필터링 요청
            entity_list = "\n".join([
                f"{i}. [{e['label']}] \"{e['text']}\" (위치: {e['start']}-{e['end']})"
                for i, e in enumerate(entities)
            ])

            filter_request = {
                "custom_id": f"filter_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4.1-mini",
                    "messages": [
                        {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                        {"role": "user", "content": f"## 텍스트\n{text}\n\n## 추출된 엔티티\n{entity_list}"}
                    ],
                    "temperature": 0,
                    "max_tokens": 500
                }
            }
            requests.append(filter_request)

        if task in ['add', 'both']:
            # 추가 요청
            existing = ", ".join([f"[{e['label']}]{e['text']}" for e in entities])

            add_request = {
                "custom_id": f"add_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4.1-mini",
                    "messages": [
                        {"role": "system", "content": ADD_SYSTEM_PROMPT},
                        {"role": "user", "content": f"## 텍스트\n{text}\n\n## 이미 추출된 엔티티\n{existing}"}
                    ],
                    "temperature": 0,
                    "max_tokens": 1000
                }
            }
            requests.append(add_request)

    # JSONL로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + '\n')

    print(f"[완료] {len(requests)}개 요청 생성: {output_file}")
    return len(requests)


def submit_batch(request_file: Path):
    """배치 요청 제출"""
    client = OpenAI()

    # 파일 업로드
    with open(request_file, 'rb') as f:
        batch_file = client.files.create(file=f, purpose="batch")

    print(f"[업로드] 파일 ID: {batch_file.id}")

    # 배치 생성
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    print(f"[제출] 배치 ID: {batch.id}")
    print(f"[상태] {batch.status}")

    return batch.id


def check_status(batch_id: str):
    """배치 상태 확인"""
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    print(f"배치 ID: {batch.id}")
    print(f"상태: {batch.status}")
    print(f"생성: {batch.created_at}")

    if batch.request_counts:
        print(f"요청: 총 {batch.request_counts.total}, "
              f"완료 {batch.request_counts.completed}, "
              f"실패 {batch.request_counts.failed}")

    if batch.output_file_id:
        print(f"출력 파일: {batch.output_file_id}")

    return batch


def download_results(batch_id: str, output_file: Path):
    """배치 결과 다운로드"""
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        print(f"[대기] 배치 미완료 상태: {batch.status}")
        return None

    if not batch.output_file_id:
        print("[오류] 출력 파일 없음")
        return None

    # 결과 다운로드
    content = client.files.content(batch.output_file_id)

    with open(output_file, 'wb') as f:
        f.write(content.read())

    print(f"[완료] 결과 저장: {output_file}")
    return output_file


def process_results(input_file: Path, result_file: Path, output_file: Path):
    """배치 결과 처리하여 최종 JSONL 생성"""

    # 원본 문서 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        docs = [json.loads(line) for line in f]

    # 배치 결과 로드
    results = {}
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result['custom_id']
            response = result.get('response', {})
            if response.get('status_code') == 200:
                content = response['body']['choices'][0]['message']['content']
                results[custom_id] = content

    # 결과 적용
    verified_docs = []
    for idx, doc in enumerate(docs):
        entities = doc.get('entities', []).copy()

        # 필터링 적용
        filter_key = f"filter_{idx}"
        if filter_key in results:
            try:
                remove_indices = json.loads(results[filter_key])
                entities = [e for i, e in enumerate(entities) if i not in remove_indices]
            except:
                pass

        # 추가 적용
        add_key = f"add_{idx}"
        if add_key in results:
            try:
                new_entities = json.loads(results[add_key])
                for e in new_entities:
                    e['source'] = 'gpt5-batch'
                    e['code'] = ''
                entities.extend(new_entities)
            except:
                pass

        doc['entities'] = entities
        doc['verified'] = True
        doc['verified_by'] = 'gpt5-batch'
        doc['verified_at'] = datetime.now().isoformat()
        verified_docs.append(doc)

    # 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in verified_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    print(f"[완료] {len(verified_docs)}건 처리: {output_file}")
    return len(verified_docs)


def main():
    parser = argparse.ArgumentParser(description='GPT-5 Batch API NER 검증')
    subparsers = parser.add_subparsers(dest='command', help='명령')

    # submit 명령
    submit_parser = subparsers.add_parser('submit', help='배치 요청 제출')
    submit_parser.add_argument('--input', '-i', required=True, help='입력 JSONL 파일')
    submit_parser.add_argument('--task', choices=['filter', 'add', 'both'], default='both')

    # status 명령
    status_parser = subparsers.add_parser('status', help='배치 상태 확인')
    status_parser.add_argument('--batch-id', '-b', required=True, help='배치 ID')

    # download 명령
    download_parser = subparsers.add_parser('download', help='결과 다운로드')
    download_parser.add_argument('--batch-id', '-b', required=True, help='배치 ID')
    download_parser.add_argument('--output', '-o', required=True, help='출력 파일')

    # process 명령
    process_parser = subparsers.add_parser('process', help='결과 처리')
    process_parser.add_argument('--input', '-i', required=True, help='원본 JSONL')
    process_parser.add_argument('--result', '-r', required=True, help='배치 결과 파일')
    process_parser.add_argument('--output', '-o', required=True, help='최종 출력')

    args = parser.parse_args()

    if args.command == 'submit':
        input_path = Path(args.input)
        request_path = input_path.parent / f"{input_path.stem}_batch_requests.jsonl"

        # 요청 파일 생성
        create_batch_requests(input_path, request_path, args.task)

        # 제출
        batch_id = submit_batch(request_path)
        print(f"\n다음 명령으로 상태 확인:")
        print(f"  python batch_verify_gpt5.py status --batch-id {batch_id}")

    elif args.command == 'status':
        check_status(args.batch_id)

    elif args.command == 'download':
        download_results(args.batch_id, Path(args.output))

    elif args.command == 'process':
        process_results(Path(args.input), Path(args.result), Path(args.output))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
