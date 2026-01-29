"""
통합 Gazetteer 구축
- Disease: KCD-9 + 상병마스터 + NCC 암 별칭 (101개)
- Drug: pharmalex_unity
- Procedure: 고시 데이터 (향후)

변경 이력:
- v1.0 (2026-01-07): 초기 버전, 15개 암 별칭 하드코딩
- v2.0 (2026-01-07): NCC 71개 암종 기반 101개 암 별칭으로 확장
                     cancer_kcd_mapping_verified.csv에서 로드
"""
import json
import csv
import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "gazetteer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 외부 데이터 루트 (gazetteer 재구축 시 scrape-hub 경로 필요)
EXTERNAL_DATA_ROOT = Path(os.environ.get("SCRAPE_HUB_ROOT", "C:/Jimin/scrape-hub"))

# 암 별칭 매핑 파일 (검증 완료)
CANCER_MAPPING_FILE = EXTERNAL_DATA_ROOT / "neo4j/data/bridges/cancer_kcd_mapping_verified.csv"


def build_disease_gazetteer():
    """Disease Gazetteer 구축 (KCD-9 + 상병마스터)"""
    print("=== Disease Gazetteer 구축 ===")

    gazetteer = {}

    # 1. KCD-9 로드
    kcd_file = EXTERNAL_DATA_ROOT / "data/kssc/kcd-9th/normalized/kcd9_full.json"
    with open(kcd_file, 'r', encoding='utf-8') as f:
        kcd_data = json.load(f)

    for item in kcd_data.get('codes', []):
        code = item.get('code', '')
        name_kr = item.get('name_kr', '').strip()
        name_en = item.get('name_en', '').strip()

        if name_kr and code:
            key = name_kr.lower()
            if key not in gazetteer:
                gazetteer[key] = {
                    "code": code,
                    "name_kr": name_kr,
                    "name_en": name_en,
                    "source": "KCD-9"
                }

    print(f"  KCD-9: {len(gazetteer):,}개")

    # 2. 상병마스터 로드
    master_file = EXTERNAL_DATA_ROOT / "data/hira_master/배포용 상병마스터_250908(2).xlsx"
    df = pd.read_excel(master_file, sheet_name='상병분류기호(완전코드)', skiprows=10, header=0)
    df.columns = ['상병기호', '한글명', '영문명', '주상병사용구분', '성별구분', '완전코드구분', '법정감염병', '상한연령', '하한연령', '양한방구분']

    added_from_master = 0
    for _, row in df.iterrows():
        code = str(row['상병기호']).strip()
        name_kr = str(row['한글명']).strip() if pd.notna(row['한글명']) else ''
        name_en = str(row['영문명']).strip() if pd.notna(row['영문명']) else ''

        if name_kr and code and name_kr != '한글명':
            key = name_kr.lower()
            if key not in gazetteer:
                gazetteer[key] = {
                    "code": code,
                    "name_kr": name_kr,
                    "name_en": name_en,
                    "source": "상병마스터"
                }
                added_from_master += 1

    print(f"  상병마스터 추가: {added_from_master:,}개")
    print(f"  총 Disease: {len(gazetteer):,}개")

    # 3. 암 별칭 추가 (NCC 71개 암종 기반, 검증 완료)
    # 근거: docs/cancer_kcd_mapping_analysis.md
    # 소스: cancer_kcd_mapping_verified.csv (KCD-9 수동 검증)
    cancer_aliases = {}

    if CANCER_MAPPING_FILE.exists():
        with open(CANCER_MAPPING_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cancer_name = row.get('cancer_name', '').strip()
                kcd_code = row.get('kcd_code', '').strip()
                kcd_name_kr = row.get('kcd_name_kr', '').strip()

                if cancer_name and kcd_code:
                    # 점(.) 제거한 코드도 지원 (C16.9 → C169)
                    code_normalized = kcd_code.replace('.', '')
                    cancer_aliases[cancer_name] = {
                        "code": code_normalized,
                        "kcd_code_original": kcd_code,
                        "kcd_name_kr": kcd_name_kr
                    }
        print(f"  암 별칭 CSV 로드: {len(cancer_aliases)}개")
    else:
        print(f"  [경고] 암 별칭 파일 없음: {CANCER_MAPPING_FILE}")

    added_cancer = 0
    for alias, info in cancer_aliases.items():
        key = alias.lower()
        if key not in gazetteer:
            gazetteer[key] = {
                "code": info["code"],
                "name_kr": alias,
                "name_en": "",
                "source": "NCC암별칭",
                "kcd_original": info.get("kcd_code_original", ""),
                "kcd_name": info.get("kcd_name_kr", "")
            }
            added_cancer += 1

    print(f"  암 별칭 Gazetteer 추가: {added_cancer}개")

    # 저장
    output_file = OUTPUT_DIR / "disease_gazetteer.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "version": "2.0",
            "total": len(gazetteer),
            "sources": ["KCD-9", "상병마스터", "NCC암별칭"],
            "cancer_aliases_count": len(cancer_aliases),
            "entries": gazetteer
        }, f, ensure_ascii=False, indent=2)

    print(f"  저장: {output_file}")

    return gazetteer


def build_drug_gazetteer():
    """Drug Gazetteer 구축 (pharmalex_unity)"""
    print("\n=== Drug Gazetteer 구축 ===")

    gazetteer = {}

    # pharmalex_unity 로드
    pharma_file = EXTERNAL_DATA_ROOT / "data/pharmalex_unity/merged_pharma_data_active.csv"

    with open(pharma_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get('제품코드', '')
            product_name = row.get('제품명', '').strip()
            generic_name = row.get('일반명', '').strip()

            # 제품명 처리
            if product_name:
                # 규격 부분 제거
                short_name = product_name.split('_')[0].split('(')[0].strip()
                key = short_name.lower()

                if key and key not in gazetteer:
                    gazetteer[key] = {
                        "code": code,
                        "product_name": product_name,
                        "generic_name": generic_name,
                        "type": "product",
                        "source": "pharmalex"
                    }

            # 일반명 처리
            if generic_name:
                key = generic_name.lower()
                if key and key not in gazetteer:
                    gazetteer[key] = {
                        "code": code,
                        "product_name": product_name,
                        "generic_name": generic_name,
                        "type": "generic",
                        "source": "pharmalex"
                    }

    print(f"  pharmalex: {len(gazetteer):,}개")

    # 저장
    output_file = OUTPUT_DIR / "drug_gazetteer.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "version": "1.0",
            "total": len(gazetteer),
            "sources": ["pharmalex_unity"],
            "entries": gazetteer
        }, f, ensure_ascii=False, indent=2)

    print(f"  저장: {output_file}")

    return gazetteer


def build_procedure_gazetteer():
    """Procedure Gazetteer 구축 (EDI 수가 데이터)

    소스: data/hins/downloads/edi/ 폴더의 SNOMED-CT 매핑 파일들
    - 9장: 처치및수술료 (핵심)
    - 2장: 검사
    - 7장: 이학요법료
    - 1장: 기본진료료
    - 3장: 영상진단및방사선
    - 18장: 응급의료수가
    """
    print("\n=== Procedure Gazetteer 구축 ===")

    gazetteer = {}
    stats = {}

    # EDI 수가 파일 목록
    EDI_DIR = EXTERNAL_DATA_ROOT / "data/hins/downloads/edi"
    procedure_files = [
        ("9장_19_20용어매핑테이블(처치및수술료)_(심평원코드-SNOMED_CT).xlsx", "처치수술"),
        ("2장_19_20용어매핑테이블(검사)_(심평원코드-SNOMED_CT).xlsx", "검사"),
        ("7장_20용어매핑테이블(이학요법료)_(심평원코드_SNOMED_CT).xlsx", "이학요법"),
        ("1장_20용어매핑테이블(기본진료료)_(심평원코드_SNOMED_CT).xlsx", "기본진료"),
        ("3장_20용어매핑테이블(영상진단및방사선)_(심평원코드_SNOMED_CT).xlsx", "영상진단"),
        ("18장_20용어매핑테이블(응급의료수가)_(심평원코드_SNOMED_CT).xlsx", "응급의료"),
    ]

    for filename, category in procedure_files:
        filepath = EDI_DIR / filename
        if not filepath.exists():
            print(f"  [SKIP] {filename}")
            continue

        try:
            df = pd.read_excel(filepath)

            # 컬럼명 정규화
            if 'term_cd' in df.columns:
                code_col = 'term_cd'
                name_col = 'term_kr'
            elif 'Source term' in df.columns:
                # 첫 행이 헤더인 경우
                df.columns = df.iloc[0]
                df = df.iloc[1:]
                code_col = '수가코드'
                name_col = '한글명'
            else:
                print(f"  [ERROR] Unknown format: {filename}")
                continue

            count = 0
            for _, row in df.iterrows():
                code = str(row.get(code_col, '')).strip()
                name = str(row.get(name_col, '')).strip()

                if name and code and name != 'nan' and code != 'nan':
                    key = name.lower()
                    if key not in gazetteer:
                        gazetteer[key] = {
                            "code": code,
                            "name": name,
                            "category": category,
                            "source": "EDI수가"
                        }
                        count += 1

            stats[category] = count
            print(f"  {category}: {count}개")

        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")

    print(f"  총 Procedure: {len(gazetteer):,}개")

    # 저장
    output_file = OUTPUT_DIR / "procedure_gazetteer.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "version": "1.0",
            "total": len(gazetteer),
            "sources": ["EDI수가"],
            "categories": list(stats.keys()),
            "stats": stats,
            "entries": gazetteer
        }, f, ensure_ascii=False, indent=2)

    print(f"  저장: {output_file}")

    return gazetteer


def main():
    print("=" * 50)
    print("통합 Gazetteer 구축")
    print("=" * 50)

    # Disease
    disease = build_disease_gazetteer()

    # Drug
    drug = build_drug_gazetteer()

    # Procedure
    procedure = build_procedure_gazetteer()

    # 요약
    print("\n" + "=" * 50)
    print("구축 완료 요약")
    print("=" * 50)
    print(f"  Disease: {len(disease):,}개")
    print(f"  Drug: {len(drug):,}개")
    print(f"  Procedure: {len(procedure):,}개")
    print(f"  총계: {len(disease) + len(drug) + len(procedure):,}개")
    print(f"\n저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
