"""
Biomarker Gazetteer 생성 스크립트

입력: neo4j/data/bridges/biomarkers_extracted.json
출력: data/gazetteer/biomarker_gazetteer.json
"""

import json
import os
from pathlib import Path
from datetime import datetime

# 경로
REPO_ROOT = Path(__file__).parent.parent.parent
EXTERNAL_DATA_ROOT = Path(os.environ.get("SCRAPE_HUB_ROOT", "C:/Jimin/scrape-hub"))
INPUT_FILE = EXTERNAL_DATA_ROOT / 'neo4j' / 'data' / 'bridges' / 'biomarkers_extracted.json'
OUTPUT_FILE = REPO_ROOT / 'data' / 'gazetteer' / 'biomarker_gazetteer.json'


def build_gazetteer():
    """바이오마커 Gazetteer 생성"""
    print("=" * 60)
    print("Biomarker Gazetteer 생성")
    print("=" * 60)

    # 입력 파일 로드
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    biomarkers = data.get('biomarkers', [])
    print(f"\n입력 바이오마커: {len(biomarkers)}개")

    entries = {}

    for bm in biomarkers:
        biomarker_id = bm['biomarker_id']
        name_en = bm['biomarker_name_en']
        name_ko = bm['biomarker_name_ko']
        bm_type = bm['biomarker_type']
        gene = bm.get('protein_gene', '')
        cancer_types = bm.get('cancer_types', [])

        info = {
            'biomarker_id': biomarker_id,
            'name_en': name_en,
            'name_ko': name_ko,
            'type': bm_type,
            'gene': gene,
            'cancer_types': cancer_types,
        }

        # 다양한 별칭 생성
        aliases = set()

        # 영문 이름 (소문자)
        aliases.add(name_en.lower())

        # 한글 이름 (소문자로 저장)
        aliases.add(name_ko.lower())

        # 영문 이름 변형
        # HER2 -> her2, her-2
        if '-' not in name_en:
            # ALK -> alk
            aliases.add(name_en.lower())
        else:
            # BCR-ABL -> bcr-abl, bcrabl
            aliases.add(name_en.lower())
            aliases.add(name_en.lower().replace('-', ''))

        # 숫자가 포함된 경우 변형
        # CDK4/6 -> cdk4/6, cdk4, cdk6
        if '/' in name_en:
            aliases.add(name_en.lower())
            parts = name_en.split('/')
            for part in parts:
                if part.strip():
                    aliases.add(part.strip().lower())
                    # CDK4 -> cdk4
                    base = ''.join(c for c in parts[0] if not c.isdigit())
                    aliases.add(f"{base}{part.strip()}".lower())

        # 한글 변형 - 일반 용어 제외
        # HER2 수용체 -> her2 (수용체는 제외)
        GENERIC_TERMS = {'수용체', '융합', '유전자', '돌연변이', '양성', '음성', '억제', '표적', '안드로겐', '에스트로겐', '프로게스테론'}
        if ' ' in name_ko:
            parts = name_ko.split()
            for part in parts:
                if len(part) >= 2 and part not in GENERIC_TERMS:
                    aliases.add(part.lower())

        # 추가 별칭 (수동 정의)
        extra_aliases = {
            'HER2': ['her2', 'her-2', 'erbb2', 'neu', 'her2/neu', 'her2 양성', 'her2양성'],
            'EGFR': ['egfr', 'erbb1', 'her1', 'egfr 돌연변이', 'egfr돌연변이'],
            'ALK': ['alk', 'alk 융합', 'alk융합', 'alk 양성', 'alk양성'],
            'BRAF': ['braf', 'braf v600e', 'brafv600e', 'braf 돌연변이', 'braf돌연변이'],
            'PD-1': ['pd-1', 'pd1', 'pdcd1'],
            'PD-L1': ['pd-l1', 'pdl1', 'cd274'],
            'ER': ['er', 'er 양성', 'er양성', '에스트로겐수용체'],
            'BRCA': ['brca', 'brca1', 'brca2', 'brca 돌연변이'],
            'KRAS': ['kras', 'kras 돌연변이', 'kras야생형'],
            'NTRK': ['ntrk', 'ntrk 융합', 'ntrk1', 'ntrk2', 'ntrk3'],
            'ROS1': ['ros1', 'ros1 융합', 'ros1융합'],
            'BCR-ABL': ['bcr-abl', 'bcrabl', 'philadelphia', '필라델피아'],
            'VEGF': ['vegf', 'vegfa', 'vegfr'],
            'mTOR': ['mtor', 'm-tor'],
            'PARP': ['parp', 'parp1', 'parp2'],
        }

        if name_en in extra_aliases:
            for alias in extra_aliases[name_en]:
                aliases.add(alias.lower())

        # 각 별칭을 엔트리로 추가
        for alias in aliases:
            if alias and len(alias) >= 2:
                entries[alias] = info

    # 추가 바이오마커 (수동 추가)
    additional_biomarkers = [
        {
            'biomarker_id': 'BIOMARKER_018',
            'name_en': 'BRCA',
            'name_ko': 'BRCA 돌연변이',
            'type': 'mutation',
            'gene': 'BRCA1/BRCA2',
            'cancer_types': ['유방암', '난소암'],
            'aliases': ['brca', 'brca1', 'brca2', 'brca 돌연변이', 'brca1/2']
        },
        {
            'biomarker_id': 'BIOMARKER_019',
            'name_en': 'KRAS',
            'name_ko': 'KRAS 돌연변이',
            'type': 'mutation',
            'gene': 'KRAS',
            'cancer_types': ['대장암', '폐암', '췌장암'],
            'aliases': ['kras', 'kras 돌연변이', 'kras 야생형', 'kras wild-type']
        },
        {
            'biomarker_id': 'BIOMARKER_020',
            'name_en': 'NTRK',
            'name_ko': 'NTRK 융합',
            'type': 'fusion_gene',
            'gene': 'NTRK1/2/3',
            'cancer_types': ['범종양'],
            'aliases': ['ntrk', 'ntrk1', 'ntrk2', 'ntrk3', 'ntrk 융합', 'trk']
        },
        {
            'biomarker_id': 'BIOMARKER_021',
            'name_en': 'MSI-H',
            'name_ko': '현미부수체 불안정성',
            'type': 'genomic_instability',
            'gene': 'MLH1/MSH2/MSH6/PMS2',
            'cancer_types': ['대장암', '위암', '자궁내막암'],
            'aliases': ['msi', 'msi-h', 'msi-high', '현미부수체', 'microsatellite instability']
        },
        {
            'biomarker_id': 'BIOMARKER_022',
            'name_en': 'TMB',
            'name_ko': '종양변이부담',
            'type': 'genomic_instability',
            'gene': '',
            'cancer_types': ['범종양'],
            'aliases': ['tmb', 'tmb-h', 'tmb-high', '종양변이부담', 'tumor mutation burden']
        },
        {
            'biomarker_id': 'BIOMARKER_023',
            'name_en': 'PR',
            'name_ko': '프로게스테론 수용체',
            'type': 'protein',
            'gene': 'PGR',
            'cancer_types': ['유방암'],
            'aliases': ['pr', 'pr 양성', 'pr양성', '프로게스테론수용체', 'progesterone receptor']
        },
    ]

    for bm in additional_biomarkers:
        info = {
            'biomarker_id': bm['biomarker_id'],
            'name_en': bm['name_en'],
            'name_ko': bm['name_ko'],
            'type': bm['type'],
            'gene': bm['gene'],
            'cancer_types': bm['cancer_types'],
        }
        for alias in bm['aliases']:
            if alias and len(alias) >= 2:
                entries[alias.lower()] = info

    # 결과 저장
    output = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'source': 'biomarkers_extracted.json + manual additions',
            'total_entries': len(entries),
            'unique_biomarkers': len(set(e['biomarker_id'] for e in entries.values())),
        },
        'entries': entries,
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n생성 완료:")
    print(f"  - 총 엔트리: {len(entries)}개")
    print(f"  - 고유 바이오마커: {output['metadata']['unique_biomarkers']}개")
    print(f"  - 출력 파일: {OUTPUT_FILE}")

    # 샘플 출력
    print("\n샘플 엔트리:")
    for key in list(entries.keys())[:10]:
        info = entries[key]
        print(f"  {key}: {info['name_en']} ({info['name_ko']})")


if __name__ == '__main__':
    build_gazetteer()
