from gliner2 import GLiNER2
import json

model = GLiNER2.from_pretrained('fastino/gliner2-base-v1')
model.load_lora_adapter('gliner2_silver/best')

test_data = [json.loads(l) for l in open('test.jsonl')]
ENTITY_DESCRIPTIONS = {
    'Disease': '질병, 증상, 의학적 상태',
    'Drug': '약물, 의약품, 치료제',
    'Procedure': '의료 시술, 수술, 검사',
    'Biomarker': '바이오마커, 검사 수치'
}

tp, fp, fn = 0, 0, 0
for doc in test_data:
    gold = set()
    for label, mentions in doc['entities'].items():
        for m in mentions:
            gold.add((m, label))
    result = model.extract_entities(doc['text'], ENTITY_DESCRIPTIONS, threshold=0.3)
    pred = set()
    for label, mentions in result.get('entities', {}).items():
        for m in mentions:
            pred.add((m, label))
    tp += len(gold & pred)
    fp += len(pred - gold)
    fn += len(gold - pred)

p = tp/(tp+fp) if tp+fp else 0
r = tp/(tp+fn) if tp+fn else 0
f1 = 2*p*r/(p+r) if p+r else 0
print(f'파인튜닝 후: P={p:.4f} R={r:.4f} F1={f1:.4f}')
print(f'베이스라인:  P=0.4808 R=0.3521 F1=0.4065')
print(f'Delta F1: {f1-0.4065:+.4f}')
