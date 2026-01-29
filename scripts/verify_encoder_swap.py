"""
GLiNER2 한국어 인코더 교체 사전 검증 스크립트

GPU 없이 로컬에서 실행하여 인코더 교체 호환성을 검증한다.

검증 항목 (7단계):
  [독립 실행 — GLiNER2 불필요]
  1. Config 클래스: DebertaV2Config 확인
  2. 토크나이저 + 스페셜 토큰: 형태소 분절, CLS/SEP/PAD/UNK id, vocab_size 일치
  3. Forward pass shape + NaN/Inf

  [GLiNER2 필요 — 미설치 시 SKIP]
  4. 히든사이즈 교차 검증: 인코더 ↔ GLiNER2 head 기대 차원
  5. loss.backward() + gradient 흐름: 더미 배치로 CPU 학습 1스텝
  6. 파라미터 그룹: encoder/head 분리가 올바른지 확인
  7. extract_entities 추론 경로: 한국어 문장 3개 최소 동작

사용법:
    python verify_encoder_swap.py
    python verify_encoder_swap.py --report  # 검증 리포트 파일 생성

종료 코드:
    0: 모든 검증 통과
    1: 하나 이상 실패
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

KOREAN_ENCODER = "team-lucid/deberta-v3-base-korean"
GLINER2_PRETRAINED = "fastino/gliner2-base-v1"

ENTITY_DESCRIPTIONS = {
    "Disease": "질병, 증상, 의학적 상태 (예: 당뇨병, 고혈압, 폐렴, 암)",
    "Drug": "약물, 의약품, 치료제 (예: 인슐린, 아스피린, 항생제)",
    "Procedure": "의료 시술, 수술, 검사 (예: 내시경, MRI, 수술)",
    "Biomarker": "바이오마커, 검사 수치, 생체 지표 (예: 혈당, 콜레스테롤, 종양표지자)",
}

TEST_SENTENCES = [
    "당뇨병 환자에게 인슐린을 투여한다",
    "고혈압 치료를 위해 혈압강하제를 처방하였다",
    "뇌졸중 후 재활치료로 물리치료를 시행한다",
    "위내시경 검사에서 위궤양이 발견되었다",
    "혈당 수치가 126mg/dL 이상으로 당뇨병 진단 기준을 충족한다",
]

# 추론 검증용 문장 (질병/약/검사 포함)
INFERENCE_SENTENCES = [
    "당뇨병 환자에게 인슐린을 투여하고 혈당을 모니터링한다",
    "폐렴 진단을 위해 흉부 CT 촬영을 시행하였다",
    "고혈압에 암로디핀 5mg을 처방하고 혈압을 측정하였다",
]

# 로그 버퍼 — 리포트 생성용
_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(msg)


def separator(title):
    log(f"\n{'=' * 60}")
    log(f"  {title}")
    log(f"{'=' * 60}")


def check_pass(name, detail=""):
    msg = f"  [PASS] {name}"
    if detail:
        msg += f"  ({detail})"
    log(msg)


def check_fail(name, detail=""):
    msg = f"  [FAIL] {name}"
    if detail:
        msg += f"\n         → {detail}"
    log(msg)
    return False


def check_skip(name, detail=""):
    msg = f"  [SKIP] {name}"
    if detail:
        msg += f"  ({detail})"
    log(msg)


# ==============================================================
# 검증 1: Config 클래스
# ==============================================================

def verify_config_class():
    """한국어 DeBERTa가 DebertaV2Config으로 로드되는지 확인"""
    separator("검증 1/7: Config 클래스")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(KOREAN_ENCODER)
    config_cls = config.__class__.__name__
    model_type = getattr(config, "model_type", "N/A")

    log(f"  모델: {KOREAN_ENCODER}")
    log(f"  Config 클래스: {config_cls}")
    log(f"  model_type: {model_type}")

    if "DebertaV2" in config_cls or "deberta-v2" in model_type:
        check_pass("DebertaV2Config 확인")
        return True, config
    else:
        return check_fail(
            "DebertaV2Config",
            f"기대: DebertaV2Config, 실제: {config_cls}"
        ), config


# ==============================================================
# 검증 2: 토크나이저 + 스페셜 토큰 + vocab_size 일치
# ==============================================================

def verify_tokenizer_and_special_tokens(korean_config):
    """토크나이저 품질, 스페셜 토큰 존재, vocab_size 일치를 종합 검증"""
    separator("검증 2/7: 토크나이저 + 스페셜 토큰 + vocab_size")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(KOREAN_ENCODER)

    log(f"  토크나이저 클래스: {tok.__class__.__name__}")
    log(f"  vocab_size (tokenizer): {tok.vocab_size}")
    log(f"  vocab_size (config):    {korean_config.vocab_size}")
    log(f"  model_max_length: {tok.model_max_length}")
    log()

    all_ok = True

    # --- 2a. vocab_size 일치 ---
    if tok.vocab_size == korean_config.vocab_size:
        check_pass(f"vocab_size 일치: tokenizer={tok.vocab_size}, config={korean_config.vocab_size}")
    else:
        delta = abs(korean_config.vocab_size - tok.vocab_size)
        # DeBERTa 모델은 config.vocab_size > tokenizer.vocab_size인 경우가 흔함
        # (여분의 embedding slot). resize_token_embeddings()으로 처리 가능.
        if delta < 200 and korean_config.vocab_size > tok.vocab_size:
            log(f"  [WARN] vocab_size 차이: tokenizer={tok.vocab_size}, config={korean_config.vocab_size} (delta={delta})")
            log(f"         → DeBERTa 정상 범위. resize_token_embeddings()로 보정됨")
        else:
            check_fail(
                "vocab_size 불일치",
                f"tokenizer={tok.vocab_size}, config={korean_config.vocab_size} (delta={delta}). "
                "resize_token_embeddings 필요"
            )
            all_ok = False

    # --- 2b. 스페셜 토큰 존재 ---
    special_tokens = {
        "cls_token": tok.cls_token,
        "sep_token": tok.sep_token,
        "pad_token": tok.pad_token,
        "unk_token": tok.unk_token,
    }

    log(f"\n  스페셜 토큰:")
    for name, token in special_tokens.items():
        if token is not None:
            token_id = tok.convert_tokens_to_ids(token)
            log(f"    {name:12} = '{token}' (id={token_id})")
            if token_id is None or (isinstance(token_id, int) and token_id < 0):
                check_fail(f"{name} id 비정상", f"token='{token}', id={token_id}")
                all_ok = False
        else:
            check_fail(f"{name} 없음", "None")
            all_ok = False

    if all(v is not None for v in special_tokens.values()):
        check_pass("CLS/SEP/PAD/UNK 스페셜 토큰 존재")

    # --- 2c. 스페셜 토큰 인코딩/디코딩 왕복 ---
    test_text = "당뇨병"
    encoded = tok(test_text, return_tensors="pt")
    input_ids = encoded["input_ids"][0].tolist()
    decoded = tok.decode(input_ids, skip_special_tokens=False)
    decoded_clean = tok.decode(input_ids, skip_special_tokens=True)

    log(f"\n  인코딩/디코딩 왕복:")
    log(f"    입력:     '{test_text}'")
    log(f"    input_ids: {input_ids}")
    log(f"    디코딩(전체): '{decoded}'")
    log(f"    디코딩(순수): '{decoded_clean}'")

    # CLS는 시작, SEP는 끝에 있어야 함
    cls_id = tok.convert_tokens_to_ids(tok.cls_token) if tok.cls_token else None
    sep_id = tok.convert_tokens_to_ids(tok.sep_token) if tok.sep_token else None

    if cls_id is not None and input_ids[0] == cls_id:
        check_pass(f"CLS 토큰이 시작 위치에 존재 (id={cls_id})")
    else:
        # DeBERTa-v3는 [CLS] 대신 다른 방식일 수 있음
        log(f"    참고: input_ids[0]={input_ids[0]}, cls_id={cls_id}")

    if sep_id is not None and input_ids[-1] == sep_id:
        check_pass(f"SEP 토큰이 끝 위치에 존재 (id={sep_id})")
    else:
        log(f"    참고: input_ids[-1]={input_ids[-1]}, sep_id={sep_id}")

    # --- 2d. 형태소 분절 품질 ---
    log(f"\n  형태소 분절 품질:")

    syllable_total = 0
    korean_token_total = 0

    for sent in TEST_SENTENCES[:3]:
        tokens = tok.tokenize(sent)
        log(f"    입력: \"{sent}\"")
        log(f"    토큰: {tokens} ({len(tokens)}개)")

        single_syllable = sum(
            1 for t in tokens
            if len(t.replace("▁", "").replace("##", "")) == 1
            and any("\uac00" <= c <= "\ud7a3" for c in t)
        )
        korean_chars = sum(
            1 for t in tokens
            if any("\uac00" <= c <= "\ud7a3" for c in t)
        )
        syllable_total += single_syllable
        korean_token_total += korean_chars

    if korean_token_total > 0:
        ratio = syllable_total / korean_token_total
        if ratio > 0.7:
            check_fail("형태소 분절", f"단일음절 비율 {ratio:.0%} — 음절 단위 분절 감지")
            all_ok = False
        else:
            check_pass(f"형태소 분절 양호 (단일음절 {ratio:.0%})")

    return all_ok, tok


# ==============================================================
# 검증 3: Forward pass shape + NaN/Inf
# ==============================================================

def verify_forward_pass():
    """인코더 단독 forward pass — shape, NaN/Inf, 통계"""
    separator("검증 3/7: Forward pass (인코더 단독)")

    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(KOREAN_ENCODER)
    tokenizer = AutoTokenizer.from_pretrained(KOREAN_ENCODER)

    model.eval()
    test_text = TEST_SENTENCES[0]

    tokens = tokenizer(
        test_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    log(f"  입력: \"{test_text}\"")
    log(f"  input_ids shape: {tokens['input_ids'].shape}")
    log(f"  토큰 ID: {tokens['input_ids'][0].tolist()}")

    with torch.no_grad():
        output = model(**tokens)

    hidden = output.last_hidden_state
    batch_size, seq_len, hidden_size = hidden.shape

    log(f"  출력 shape: {hidden.shape}")
    log(f"  기대: (1, seq_len, 768)")

    all_ok = True

    if batch_size == 1 and hidden_size == 768 and seq_len > 1:
        check_pass(f"shape 정상: batch={batch_size}, seq={seq_len}, hidden={hidden_size}")
    else:
        check_fail("shape 이상", f"실제: ({batch_size}, {seq_len}, {hidden_size})")
        all_ok = False

    if torch.isnan(hidden).any():
        check_fail("NaN 감지", "출력에 NaN 존재")
        all_ok = False
    elif torch.isinf(hidden).any():
        check_fail("Inf 감지", "출력에 Inf 존재")
        all_ok = False
    else:
        check_pass("출력 값 정상 (NaN/Inf 없음)")

    log(f"\n  출력 통계:")
    log(f"    mean: {hidden.mean().item():.6f}")
    log(f"    std:  {hidden.std().item():.6f}")
    log(f"    min:  {hidden.min().item():.6f}")
    log(f"    max:  {hidden.max().item():.6f}")

    return all_ok


# ==============================================================
# 검증 4: 히든사이즈 교차 검증 (GLiNER2 head 기대 차원)
# ==============================================================

def verify_hidden_size_cross(gliner2_model, korean_config):
    """인코더 hidden_size가 GLiNER2 head가 기대하는 입력 차원과 맞는지 확인"""
    separator("검증 4/7: 히든사이즈 교차 검증 (인코더 ↔ head)")

    import torch

    korean_hidden = korean_config.hidden_size
    log(f"  한국어 인코더 hidden_size: {korean_hidden}")

    # GLiNER2 내부 인코더의 hidden_size 확인
    gliner2_encoder_hidden = None
    for name, module in gliner2_model.named_modules():
        cls_name = module.__class__.__name__
        if "DebertaV2" in cls_name and hasattr(module, "config"):
            gliner2_encoder_hidden = module.config.hidden_size
            log(f"  GLiNER2 기존 인코더 hidden_size: {gliner2_encoder_hidden}")
            log(f"  인코더 경로: {name}")
            break

    if gliner2_encoder_hidden is None:
        # fallback: config.hidden_size
        if hasattr(gliner2_model, "config") and hasattr(gliner2_model.config, "hidden_size"):
            gliner2_encoder_hidden = gliner2_model.config.hidden_size
            log(f"  GLiNER2 config.hidden_size: {gliner2_encoder_hidden}")

    if gliner2_encoder_hidden is None:
        check_fail("GLiNER2 인코더 hidden_size를 찾을 수 없음")
        return False

    # head의 첫 Linear 레이어 입력 차원 탐색
    head_input_dims = []
    for name, module in gliner2_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 인코더 내부 Linear는 제외
            if any(kw in name.lower() for kw in ["deberta", "bert_layer", "encoder"]):
                continue
            if module.in_features == gliner2_encoder_hidden:
                head_input_dims.append((name, module.in_features, module.out_features))

    if head_input_dims:
        log(f"\n  Head Linear 레이어 (in_features == encoder hidden):")
        for name, in_f, out_f in head_input_dims[:5]:
            log(f"    {name}: in={in_f}, out={out_f}")

    # 교차 검증
    all_ok = True

    if korean_hidden == gliner2_encoder_hidden:
        check_pass(f"hidden_size 일치: 한국어={korean_hidden}, GLiNER2={gliner2_encoder_hidden}")
    else:
        check_fail(
            "hidden_size 불일치",
            f"한국어={korean_hidden}, GLiNER2={gliner2_encoder_hidden}. "
            "head 가중치 재사용 불가"
        )
        all_ok = False

    if head_input_dims:
        mismatched = [h for h in head_input_dims if h[1] != korean_hidden]
        if not mismatched:
            check_pass("Head Linear 레이어 입력 차원 호환")
        else:
            for name, in_f, out_f in mismatched:
                check_fail(f"Head '{name}' 입력 차원 불일치", f"in={in_f}, 기대={korean_hidden}")
                all_ok = False

    return all_ok


# ==============================================================
# 검증 5: loss.backward() + gradient 흐름
# ==============================================================

def verify_loss_backward_and_gradients(gliner2_model, encoder_path):
    """더미 배치로 loss.backward() 수행 후 gradient가 인코더에 흐르는지 확인"""
    separator("검증 5/7: loss.backward() + gradient 흐름")

    import torch

    # swap_encoder 에서 이미 교체된 모델을 받음
    model = gliner2_model
    model.train()

    # 더미 InputExample 생성
    try:
        from gliner2.training.data import InputExample
        dummy_batch = [
            InputExample(
                text="당뇨병 환자에게 인슐린을 투여한다",
                entities={"Disease": ["당뇨병"], "Drug": ["인슐린"]},
                entity_descriptions=ENTITY_DESCRIPTIONS,
            ),
            InputExample(
                text="고혈압 치료를 위해 혈압강하제를 처방하였다",
                entities={"Disease": ["고혈압"], "Drug": ["혈압강하제"]},
                entity_descriptions=ENTITY_DESCRIPTIONS,
            ),
        ]
    except ImportError:
        check_skip("InputExample import 실패 — gliner2.training.data 없음")
        return None

    # zero grad
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    # loss 계산 시도 (3-way fallback)
    loss = None

    # 방식 1: compute_loss
    if hasattr(model, "compute_loss"):
        try:
            loss = model.compute_loss(dummy_batch)
            if isinstance(loss, torch.Tensor):
                log(f"  compute_loss() 성공 → loss={loss.item():.4f}")
        except Exception as e:
            log(f"  compute_loss() 실패: {e}")
            loss = None

    # 방식 2: train_step
    if loss is None and hasattr(model, "train_step"):
        try:
            result = model.train_step(dummy_batch)
            if isinstance(result, torch.Tensor):
                loss = result
            elif isinstance(result, dict) and "loss" in result:
                loss = result["loss"]
            if loss is not None:
                log(f"  train_step() 성공 → loss={loss.item():.4f}")
        except Exception as e:
            log(f"  train_step() 실패: {e}")
            loss = None

    # 방식 3: forward(examples=...)
    if loss is None:
        try:
            result = model(examples=dummy_batch)
            if isinstance(result, torch.Tensor):
                loss = result
            elif isinstance(result, dict) and "loss" in result:
                loss = result["loss"]
            if loss is not None:
                log(f"  forward(examples=...) 성공 → loss={loss.item():.4f}")
        except Exception as e:
            log(f"  forward(examples=...) 실패: {e}")

    if loss is None:
        check_fail("loss 계산 실패", "compute_loss / train_step / forward 모두 실패")
        return False

    # NaN/Inf 체크
    if torch.isnan(loss):
        check_fail("loss가 NaN")
        return False
    if torch.isinf(loss):
        check_fail("loss가 Inf")
        return False

    check_pass(f"loss 계산 성공: {loss.item():.4f}")

    # backward
    try:
        loss.backward()
        check_pass("loss.backward() 성공")
    except Exception as e:
        check_fail("loss.backward() 실패", str(e))
        return False

    # gradient 흐름 확인
    encoder_prefix = ".".join(encoder_path) + "."
    encoder_with_grad = 0
    encoder_no_grad = 0
    encoder_zero_grad = 0
    head_with_grad = 0
    head_no_grad = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_encoder = (
            name.startswith(encoder_prefix) or
            name.startswith("encoder.") or
            any(kw in name.lower() for kw in ["deberta", "bert_layer"])
        )

        if param.grad is not None:
            if param.grad.abs().sum().item() > 0:
                if is_encoder:
                    encoder_with_grad += 1
                else:
                    head_with_grad += 1
            else:
                if is_encoder:
                    encoder_zero_grad += 1
        else:
            if is_encoder:
                encoder_no_grad += 1
            else:
                head_no_grad += 1

    log(f"\n  Gradient 분포:")
    log(f"    인코더: grad 있음={encoder_with_grad}, 0-grad={encoder_zero_grad}, grad 없음={encoder_no_grad}")
    log(f"    헤드:   grad 있음={head_with_grad}, grad 없음={head_no_grad}")

    all_ok = True

    if encoder_with_grad > 0:
        check_pass(f"인코더에 gradient 흐름 확인 ({encoder_with_grad}개 파라미터)")
    else:
        check_fail("인코더에 gradient 없음", "인코더 파라미터에 gradient가 전달되지 않음")
        all_ok = False

    if head_with_grad > 0:
        check_pass(f"헤드에 gradient 흐름 확인 ({head_with_grad}개 파라미터)")
    else:
        check_fail("헤드에 gradient 없음")
        all_ok = False

    # gradient 크기 상위 5개 출력 (인코더)
    encoder_grad_norms = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        is_encoder = (
            name.startswith(encoder_prefix) or
            name.startswith("encoder.") or
            any(kw in name.lower() for kw in ["deberta", "bert_layer"])
        )
        if is_encoder:
            encoder_grad_norms.append((name, param.grad.norm().item()))

    if encoder_grad_norms:
        encoder_grad_norms.sort(key=lambda x: x[1], reverse=True)
        log(f"\n  인코더 gradient 상위 5개:")
        for name, norm in encoder_grad_norms[:5]:
            log(f"    {name}: grad_norm={norm:.6f}")

    # cleanup
    model.zero_grad()

    return all_ok


# ==============================================================
# 검증 6: 파라미터 그룹 분리 (encoder vs head)
# ==============================================================

def verify_parameter_groups(gliner2_model, encoder_path):
    """train 스크립트의 get_parameter_groups 로직으로 encoder/head 분리를 검증"""
    separator("검증 6/7: 파라미터 그룹 분리 (encoder vs head)")

    encoder_prefix = ".".join(encoder_path) + "."

    encoder_params = []
    head_params = []
    encoder_param_names = []
    head_param_names = []

    for name, param in gliner2_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(encoder_prefix) or name.startswith("encoder."):
            encoder_params.append(param)
            encoder_param_names.append(name)
        else:
            head_params.append(param)
            head_param_names.append(name)

    # fallback: prefix 매칭 실패 시 키워드 기반
    if not encoder_params:
        log("  ⚠ prefix 매칭 실패 — 키워드 기반 fallback")
        encoder_params = []
        head_params = []
        encoder_param_names = []
        head_param_names = []

        for name, param in gliner2_model.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name.lower() for kw in ["deberta", "bert_layer"]):
                encoder_params.append(param)
                encoder_param_names.append(name)
            else:
                head_params.append(param)
                head_param_names.append(name)

    encoder_numel = sum(p.numel() for p in encoder_params)
    head_numel = sum(p.numel() for p in head_params)
    total_numel = encoder_numel + head_numel

    log(f"  인코더 파라미터: {len(encoder_params)}개 텐서, {encoder_numel:,}개 값")
    log(f"  헤드 파라미터:   {len(head_params)}개 텐서, {head_numel:,}개 값")
    log(f"  합계:            {len(encoder_params) + len(head_params)}개 텐서, {total_numel:,}개 값")

    if total_numel > 0:
        log(f"  인코더 비율: {encoder_numel / total_numel:.1%}")
        log(f"  헤드 비율:   {head_numel / total_numel:.1%}")

    all_ok = True

    if len(encoder_params) == 0:
        check_fail("인코더 파라미터 0개", "차등 학습률 적용 불가")
        all_ok = False
    else:
        check_pass(f"인코더 파라미터 {len(encoder_params)}개 확인")

    if len(head_params) == 0:
        check_fail("헤드 파라미터 0개", "헤드 학습 불가")
        all_ok = False
    else:
        check_pass(f"헤드 파라미터 {len(head_params)}개 확인")

    # 인코더 파라미터 샘플 출력
    log(f"\n  인코더 파라미터 (상위 5):")
    for name in encoder_param_names[:5]:
        log(f"    {name}")
    if len(encoder_param_names) > 5:
        log(f"    ... 외 {len(encoder_param_names) - 5}개")

    log(f"\n  헤드 파라미터 (상위 5):")
    for name in head_param_names[:5]:
        log(f"    {name}")
    if len(head_param_names) > 5:
        log(f"    ... 외 {len(head_param_names) - 5}개")

    # 미분류 파라미터 확인
    total_trainable = sum(
        1 for _, p in gliner2_model.named_parameters() if p.requires_grad
    )
    classified = len(encoder_params) + len(head_params)
    if classified < total_trainable:
        unclassified = total_trainable - classified
        check_fail(f"미분류 파라미터 {unclassified}개 존재")
        all_ok = False
    else:
        check_pass("모든 trainable 파라미터 분류 완료")

    return all_ok


# ==============================================================
# 검증 7: extract_entities 추론 경로
# ==============================================================

def verify_extract_entities(gliner2_model):
    """extract_entities로 한국어 문장 3개를 추론하여 최소 동작을 확인"""
    separator("검증 7/7: extract_entities 추론 경로")

    all_ok = True

    for i, sent in enumerate(INFERENCE_SENTENCES, 1):
        log(f"\n  [{i}/{len(INFERENCE_SENTENCES)}] \"{sent}\"")

        try:
            result = gliner2_model.extract_entities(
                sent, ENTITY_DESCRIPTIONS, threshold=0.3
            )
        except Exception as e:
            check_fail(f"문장 {i} 추론 실패", str(e))
            all_ok = False
            continue

        # 결과 타입 확인
        if not isinstance(result, dict):
            check_fail(f"문장 {i} 결과 타입 이상", f"기대: dict, 실제: {type(result).__name__}")
            all_ok = False
            continue

        # entities 키 존재 확인
        entities = result.get("entities", result)
        if not isinstance(entities, dict):
            check_fail(f"문장 {i} entities 구조 이상", f"type={type(entities).__name__}")
            all_ok = False
            continue

        # 결과 출력
        entity_count = sum(len(v) for v in entities.values() if isinstance(v, list))
        log(f"    결과: {entity_count}개 엔티티")
        for label, mentions in entities.items():
            if isinstance(mentions, list) and mentions:
                log(f"      {label}: {mentions}")

        # 라벨 이름 유효성
        for label in entities.keys():
            if label not in ENTITY_DESCRIPTIONS:
                check_fail(f"문장 {i} 알 수 없는 라벨", f"'{label}' ∉ {list(ENTITY_DESCRIPTIONS.keys())}")
                all_ok = False

    if all_ok:
        check_pass("extract_entities 추론 경로 정상")

    return all_ok


# ==============================================================
# GLiNER2 통합 검증 (검증 4~7)
# ==============================================================

def run_gliner2_checks(korean_config):
    """GLiNER2 패키지가 있을 때만 실행되는 검증 4~7"""

    try:
        from gliner2 import GLiNER2
    except ImportError:
        separator("검증 4~7: GLiNER2 의존 (SKIP)")
        log("  gliner2 패키지가 설치되지 않아 검증 4~7을 건너뜁니다.")
        log("  → pip install gliner2 후 재실행하거나 RunPod에서 실행하세요.")
        return {"hidden_cross": None, "loss_backward": None, "param_groups": None, "inference": None}

    log(f"\n  GLiNER2 로드 중: {GLINER2_PRETRAINED}")
    try:
        model = GLiNER2.from_pretrained(GLINER2_PRETRAINED)
    except Exception as e:
        separator("검증 4~7: GLiNER2 로드 실패")
        log(f"  에러: {e}")
        return {"hidden_cross": False, "loss_backward": None, "param_groups": None, "inference": None}

    results = {}

    # --- 검증 4: 히든사이즈 교차 검증 (교체 전) ---
    try:
        results["hidden_cross"] = verify_hidden_size_cross(model, korean_config)
    except Exception as e:
        log(f"  [ERROR] 검증 4: {e}")
        results["hidden_cross"] = False

    # --- 인코더 교체 수행 ---
    separator("인코더 교체 수행 (검증 5~7 준비)")

    from transformers import AutoModel, AutoTokenizer, AutoConfig
    import torch

    # find encoder path
    try:
        # find_encoder_attr 로직 인라인
        encoder_path = None
        old_encoder = None
        candidates = [
            ("model", "encoder"),
            ("encoder",),
            ("model", "token_rep_layer", "bert_layer", "bert"),
            ("model", "token_rep_layer"),
        ]
        for path in candidates:
            obj = model
            found = True
            for attr in path:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    found = False
                    break
            if found and hasattr(obj, "config"):
                encoder_path = path
                old_encoder = obj
                break

        if encoder_path is None:
            for name, module in model.named_modules():
                cls_name = module.__class__.__name__
                if "DebertaV2" in cls_name and hasattr(module, "config"):
                    encoder_path = tuple(name.split("."))
                    old_encoder = module
                    break

        if encoder_path is None:
            check_fail("인코더 경로를 찾을 수 없음")
            return {"hidden_cross": results.get("hidden_cross"), "loss_backward": False,
                    "param_groups": False, "inference": False}

        log(f"  인코더 경로: {'.'.join(encoder_path)}")
        log(f"  기존 hidden_size: {old_encoder.config.hidden_size}")

        # 교체
        korean_encoder = AutoModel.from_pretrained(KOREAN_ENCODER)
        korean_tokenizer = AutoTokenizer.from_pretrained(KOREAN_ENCODER)

        # set nested attr
        obj = model
        for attr in encoder_path[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, encoder_path[-1], korean_encoder)

        # 토크나이저 교체
        if hasattr(model, "tokenizer"):
            model.tokenizer = korean_tokenizer
        elif hasattr(model, "_tokenizer"):
            model._tokenizer = korean_tokenizer

        # 임베딩 리사이즈
        _, new_enc = None, None
        obj = model
        for attr in encoder_path:
            obj = getattr(obj, attr)
        new_enc = obj
        if hasattr(new_enc, "resize_token_embeddings"):
            new_enc.resize_token_embeddings(len(korean_tokenizer))
        elif hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(korean_tokenizer))

        log(f"  인코더 교체 완료: {KOREAN_ENCODER}")
        log(f"  토크나이저 vocab: {len(korean_tokenizer)}")

    except Exception as e:
        check_fail("인코더 교체 실패", str(e))
        import traceback
        traceback.print_exc()
        return {"hidden_cross": results.get("hidden_cross"), "loss_backward": False,
                "param_groups": False, "inference": False}

    # --- 검증 5: loss backward + gradient ---
    try:
        results["loss_backward"] = verify_loss_backward_and_gradients(model, encoder_path)
    except Exception as e:
        log(f"  [ERROR] 검증 5: {e}")
        import traceback
        traceback.print_exc()
        results["loss_backward"] = False

    # --- 검증 6: 파라미터 그룹 ---
    try:
        results["param_groups"] = verify_parameter_groups(model, encoder_path)
    except Exception as e:
        log(f"  [ERROR] 검증 6: {e}")
        results["param_groups"] = False

    # --- 검증 7: extract_entities ---
    try:
        model.eval()
        results["inference"] = verify_extract_entities(model)
    except Exception as e:
        log(f"  [ERROR] 검증 7: {e}")
        import traceback
        traceback.print_exc()
        results["inference"] = False

    return results


# ==============================================================
# 메인
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="GLiNER2 한국어 인코더 교체 사전 검증")
    parser.add_argument("--report", action="store_true", help="검증 리포트 파일 생성")
    parser.add_argument("--report-dir", default=".", help="리포트 저장 디렉토리")
    args = parser.parse_args()

    log("=" * 60)
    log("  GLiNER2 한국어 인코더 교체 사전 검증 (7-Step)")
    log("=" * 60)
    log(f"\n  한국어 인코더: {KOREAN_ENCODER}")
    log(f"  GLiNER2 모델:  {GLINER2_PRETRAINED}")
    log(f"  실행 시각:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # --- 검증 1: Config 클래스 ---
    try:
        passed, config = verify_config_class()
        results["1_config_class"] = passed
    except Exception as e:
        log(f"  [ERROR] {e}")
        results["1_config_class"] = False
        config = None

    # --- 검증 2: 토크나이저 + 스페셜 토큰 ---
    if config:
        try:
            passed, tokenizer = verify_tokenizer_and_special_tokens(config)
            results["2_tokenizer_special"] = passed
        except Exception as e:
            log(f"  [ERROR] {e}")
            results["2_tokenizer_special"] = False
    else:
        results["2_tokenizer_special"] = False

    # --- 검증 3: Forward pass ---
    try:
        results["3_forward_pass"] = verify_forward_pass()
    except Exception as e:
        log(f"  [ERROR] {e}")
        results["3_forward_pass"] = False

    # --- 검증 4~7: GLiNER2 의존 ---
    if config:
        gliner2_results = run_gliner2_checks(config)
        results["4_hidden_cross"] = gliner2_results.get("hidden_cross")
        results["5_loss_backward"] = gliner2_results.get("loss_backward")
        results["6_param_groups"] = gliner2_results.get("param_groups")
        results["7_inference"] = gliner2_results.get("inference")
    else:
        results["4_hidden_cross"] = None
        results["5_loss_backward"] = None
        results["6_param_groups"] = None
        results["7_inference"] = None

    # ==============================================================
    # 종합 결과
    # ==============================================================
    separator("종합 결과")

    labels = {
        "1_config_class":      "Config 클래스 (DebertaV2)",
        "2_tokenizer_special": "토크나이저 + 스페셜 토큰",
        "3_forward_pass":      "Forward pass (인코더 단독)",
        "4_hidden_cross":      "히든사이즈 교차 검증",
        "5_loss_backward":     "loss.backward() + gradient",
        "6_param_groups":      "파라미터 그룹 분리",
        "7_inference":         "extract_entities 추론",
    }

    has_fail = False
    has_skip = False
    for key, label in labels.items():
        val = results[key]
        if val is None:
            status = "SKIP"
            has_skip = True
        elif val:
            status = "PASS"
        else:
            status = "FAIL"
            has_fail = True
        log(f"  {status:4s} | {label}")

    log()
    if not has_fail and not has_skip:
        log("  ★ 모든 검증 통과 — 인코더 교체 + 학습 진행 가능")
    elif not has_fail and has_skip:
        log("  ○ 실행 가능 검증 모두 통과 (SKIP 항목은 GLiNER2 설치 후 확인 필요)")
    else:
        failed = [labels[k] for k, v in results.items() if v is False]
        log(f"  ⚠ 실패 항목:")
        for f in failed:
            log(f"    - {f}")
        log("  → 실패 항목을 해결 후 다시 실행하세요")

    # 리포트 파일 생성
    if args.report:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"verify_encoder_swap_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(_log_lines))
        log(f"\n  리포트 저장: {report_path}")

    return 1 if has_fail else 0


if __name__ == "__main__":
    sys.exit(main())
