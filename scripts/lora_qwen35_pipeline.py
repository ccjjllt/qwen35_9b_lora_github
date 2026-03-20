from __future__ import annotations

import argparse
import difflib
import inspect
import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import QUAD_COLS, discover_data_paths, normalize_label_df, read_csv_utf8, save_csv_no_header
from src.metrics import score_quadruples


def normalize_hf_env() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "3600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    if os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "").strip().lower() in {"1", "true", "yes"}:
        try:
            import hf_transfer  # type: ignore  # noqa: F401
        except Exception:
            print("[WARN] HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is missing; fallback to 0.")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def parse_bool(value: str, default: bool) -> bool:
    v = value.strip().lower()
    if not v:
        return default
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def resolve_hf_token(hub_cfg: dict[str, Any]) -> str | None:
    token = str(hub_cfg.get("token", "")).strip()
    if token:
        return token
    for env_key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        v = os.getenv(env_key, "").strip()
        if v:
            return v
    return None


def ensure_local_model_exists(model_name: str, local_files_only: bool) -> None:
    if not local_files_only:
        return
    p = Path(model_name)
    if not p.exists():
        raise FileNotFoundError(
            f"local_files_only=true but model path does not exist: {p}. "
            "Please upload model folder first or set model.model_name correctly."
        )


def resolve_precision(train_cfg: dict[str, Any]) -> tuple[bool, bool]:
    use_fp16 = bool(train_cfg.get("fp16", False))
    use_bf16 = bool(train_cfg.get("bf16", True))
    if use_bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("[WARN] bf16 not supported on this GPU, fallback to fp16.")
        use_bf16 = False
        use_fp16 = True
    return use_fp16, use_bf16


def filter_kwargs_for_callable(func: Any, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys())
    use = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted([k for k in kwargs.keys() if k not in accepted])
    return use, dropped


def build_training_args(
    out_dir: Path,
    resume_from: str | None,
    train_cfg: dict[str, Any],
    seed: int,
    use_fp16: bool,
    use_bf16: bool,
) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": int(train_cfg["batch_size"]),
        "per_device_eval_batch_size": int(train_cfg["eval_batch_size"]),
        "gradient_accumulation_steps": int(train_cfg["grad_accum"]),
        "num_train_epochs": float(train_cfg["num_epochs"]),
        "learning_rate": float(train_cfg["learning_rate"]),
        "weight_decay": float(train_cfg["weight_decay"]),
        "warmup_ratio": float(train_cfg["warmup_ratio"]),
        "logging_steps": int(train_cfg["logging_steps"]),
        "save_strategy": "epoch",
        "fp16": use_fp16,
        "bf16": use_bf16,
        "optim": str(train_cfg.get("optim", "paged_adamw_8bit")),
        "save_total_limit": int(train_cfg.get("save_total_limit", 2)),
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "seed": int(seed),
    }

    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())
    if "overwrite_output_dir" in params:
        kwargs["overwrite_output_dir"] = resume_from is None
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "no"
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = "no"

    filtered_kwargs, dropped = filter_kwargs_for_callable(TrainingArguments.__init__, kwargs)
    if dropped:
        print(f"[WARN] TrainingArguments unsupported keys dropped: {dropped}")
    return TrainingArguments(**filtered_kwargs)


def build_trainer(model, args: TrainingArguments, train_ds, collator, tokenizer):
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "data_collator": collator,
        "tokenizer": tokenizer,
    }
    sig = inspect.signature(Trainer.__init__)
    params = set(sig.parameters.keys())
    if "tokenizer" not in params and "processing_class" in params:
        kwargs["processing_class"] = tokenizer
        kwargs.pop("tokenizer", None)
    filtered_kwargs, dropped = filter_kwargs_for_callable(Trainer.__init__, kwargs)
    if dropped:
        print(f"[WARN] Trainer unsupported keys dropped: {dropped}")
    return Trainer(**filtered_kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    return str(text).replace("\r", " ").strip()


def build_prompt(review_text: str, categories: list[str], polarities: list[str], prompt_style: str = "en_v1") -> str:
    categories_str = ", ".join(categories)
    polarities_str = ", ".join(polarities)
    if prompt_style == "zh_v2":
        return (
            "你是电商评论观点抽取助手。请从给定评论中抽取全部四元组。\n"
            "输出必须是 JSON 数组，不能输出任何解释文字。\n"
            "每个元素字段固定为: "
            '{"AspectTerms":"...","OpinionTerms":"...","Categories":"...","Polarities":"..."}\n'
            f"Categories 只能取: {categories_str}\n"
            f"Polarities 只能取: {polarities_str}\n"
            "AspectTerms/OpinionTerms 尽量使用评论原文中的词，不要改写。\n"
            '如果没有有效四元组，输出 [{"AspectTerms":"_","OpinionTerms":"_","Categories":"_","Polarities":"_"}]。\n'
            f"评论: {review_text}\n"
            "JSON:"
        )
    return (
        "Task: extract all sentiment quadruples from a Chinese e-commerce review.\n"
        "Output format: JSON array only.\n"
        "Each item must be: "
        '{"AspectTerms":"...","OpinionTerms":"...","Categories":"...","Polarities":"..."}\n'
        f"Allowed Categories: {categories_str}\n"
        f"Allowed Polarities: {polarities_str}\n"
        'If no valid quadruple, output [{"AspectTerms":"_","OpinionTerms":"_","Categories":"_","Polarities":"_"}].\n'
        f"Review: {review_text}\n"
        "JSON:"
    )


def normalize_term(term: str) -> str:
    t = str(term).strip().replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip("`'\"[]【】()（）<>《》,，。；;：:!！？? ")
    return t if t else "_"


def nearest_label(value: str, allowed: list[str], allowed_set: set[str], cutoff: float = 0.58) -> str:
    v = normalize_term(value)
    if v in allowed_set:
        return v
    if v == "_":
        return "_"
    matched = difflib.get_close_matches(v, allowed, n=1, cutoff=cutoff)
    return matched[0] if matched else "_"


def format_target(group: pd.DataFrame) -> str:
    if group.empty:
        rows = [{"AspectTerms": "_", "OpinionTerms": "_", "Categories": "_", "Polarities": "_"}]
    else:
        rows = (
            group[QUAD_COLS]
            .astype(str)
            .drop_duplicates()
            .sort_values(["AspectTerms", "OpinionTerms", "Categories", "Polarities"])
            .to_dict("records")
        )
    return json.dumps(rows, ensure_ascii=False)


def build_records(
    reviews_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    categories: list[str],
    polarities: list[str],
    prompt_style: str = "en_v1",
) -> list[dict[str, Any]]:
    grouped = {int(i): g.copy() for i, g in labels_df.groupby("id")}
    records: list[dict[str, Any]] = []
    for row in reviews_df.itertuples(index=False):
        rid = int(row.id)
        review_text = clean_text(row.Reviews)
        prompt = build_prompt(review_text, categories=categories, polarities=polarities, prompt_style=prompt_style)
        target = format_target(grouped.get(rid, pd.DataFrame(columns=QUAD_COLS)))
        records.append({"id": rid, "prompt": prompt, "target": target})
    return records


class PromptTargetDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer, max_length: int):
        self.items: list[dict[str, Any]] = []
        eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
        for rec in records:
            prompt_ids = tokenizer(rec["prompt"], add_special_tokens=False).input_ids
            target_ids = tokenizer(rec["target"], add_special_tokens=False).input_ids
            if eos_id is not None:
                target_ids = target_ids + [eos_id]

            # Keep the answer tokens as much as possible.
            max_prompt_len = max(8, max_length - len(target_ids))
            prompt_ids = prompt_ids[:max_prompt_len]
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids

            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            attention_mask = [1] * len(input_ids)
            self.items.append(
                {
                    "id": int(rec["id"]),
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


@dataclass
class DataCollatorForCausalLM:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_id = int(self.tokenizer.pad_token_id)
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids = []
        labels = []
        attention_mask = []
        for x in features:
            n = len(x["input_ids"])
            p = max_len - n
            input_ids.append(x["input_ids"] + [pad_id] * p)
            labels.append(x["labels"] + [-100] * p)
            attention_mask.append(x["attention_mask"] + [0] * p)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def build_bnb_config(dtype: torch.dtype) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def load_lora_train_model(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    use_bf16: bool,
    local_files_only: bool,
    hf_token: str | None,
):
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
            token=hf_token,
        )
    except Exception as exc:
        raise RuntimeError(
            "Tokenizer load failed. Check model path / HF network / token. "
            "If model type qwen3_5 is unsupported, upgrade transformers via "
            "`python -m pip install -U \"git+https://github.com/huggingface/transformers.git\"`."
        ) from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=build_bnb_config(dtype=dtype),
            device_map="auto",
            trust_remote_code=True,
            local_files_only=local_files_only,
            token=hf_token,
        )
    except ValueError as exc:
        msg = str(exc)
        if "qwen3_5" in msg:
            raise RuntimeError(
                "Current transformers does not support model_type=qwen3_5. "
                "Run: python -m pip install -U \"git+https://github.com/huggingface/transformers.git\""
            ) from exc
        raise
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer


def load_lora_infer_model(
    model_name: str,
    adapter_dir: Path,
    use_bf16: bool,
    local_files_only: bool,
    hf_token: str | None,
):
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
            token=hf_token,
        )
    except Exception as exc:
        raise RuntimeError(
            "Tokenizer load failed in inference. Check model path / HF network / token."
        ) from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=build_bnb_config(dtype=dtype),
            device_map="auto",
            trust_remote_code=True,
            local_files_only=local_files_only,
            token=hf_token,
        )
    except ValueError as exc:
        msg = str(exc)
        if "qwen3_5" in msg:
            raise RuntimeError(
                "Current transformers does not support model_type=qwen3_5. "
                "Run: python -m pip install -U \"git+https://github.com/huggingface/transformers.git\""
            ) from exc
        raise
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.eval()
    return model, tokenizer


def extract_json_array(text: str) -> list[dict[str, str]]:
    if not text:
        return []
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start < 0 or end < 0 or end <= start:
        return []
    piece = cleaned[start : end + 1]
    try:
        obj = json.loads(piece)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: list[dict[str, str]] = []
    for x in obj:
        if not isinstance(x, dict):
            continue
        out.append(
            {
                "AspectTerms": str(x.get("AspectTerms", "_")).strip() or "_",
                "OpinionTerms": str(x.get("OpinionTerms", "_")).strip() or "_",
                "Categories": str(x.get("Categories", "_")).strip() or "_",
                "Polarities": str(x.get("Polarities", "_")).strip() or "_",
            }
        )
    return out


def sanitize_row(
    x: dict[str, str],
    category_list: list[str],
    polarity_list: list[str],
    category_set: set[str],
    polarity_set: set[str],
) -> dict[str, str]:
    aspect = normalize_term(x["AspectTerms"])
    opinion = normalize_term(x["OpinionTerms"])
    category = nearest_label(x["Categories"], allowed=category_list, allowed_set=category_set)
    polarity = nearest_label(x["Polarities"], allowed=polarity_list, allowed_set=polarity_set)
    return {
        "AspectTerms": aspect,
        "OpinionTerms": opinion,
        "Categories": category,
        "Polarities": polarity,
    }


@torch.no_grad()
def predict_reviews(
    model,
    tokenizer,
    reviews_df: pd.DataFrame,
    categories: list[str],
    polarities: list[str],
    category_set: set[str],
    polarity_set: set[str],
    max_prompt_length: int,
    max_new_tokens: int,
    batch_size: int,
    do_sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.9,
    num_beams: int = 1,
    num_return_sequences: int = 1,
    repetition_penalty: float = 1.0,
    min_votes: int = 1,
    progress_log_interval: int = 50,
    prompt_style: str = "en_v1",
) -> pd.DataFrame:
    prompts: list[str] = []
    ids: list[int] = []
    for row in reviews_df.itertuples(index=False):
        ids.append(int(row.id))
        prompts.append(
            build_prompt(
                clean_text(row.Reviews),
                categories=categories,
                polarities=polarities,
                prompt_style=prompt_style,
            )
        )

    rows: list[tuple[int, str, str, str, str]] = []
    total = len(prompts)
    n_return = max(1, int(num_return_sequences))
    n_beams = max(1, int(num_beams))
    if not do_sample and n_return > 1:
        n_beams = max(n_beams, n_return)
    vote_threshold = max(1, int(min_votes))

    for i in range(0, len(prompts), batch_size):
        p = prompts[i : i + batch_size]
        rid = ids[i : i + batch_size]
        enc = tokenizer(
            p,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        gen_kwargs = dict(
            **enc,
            do_sample=bool(do_sample),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=n_beams,
            num_return_sequences=n_return,
            repetition_penalty=float(repetition_penalty),
        )
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)
        out = model.generate(**gen_kwargs)

        for b in range(len(rid)):
            prompt_len = int(enc["attention_mask"][b].sum().item())
            cand_counter: Counter[tuple[str, str, str, str]] = Counter()
            base = b * n_return
            for r in range(n_return):
                row_idx = base + r
                gen_ids = out[row_idx, prompt_len:].tolist()
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                parsed = extract_json_array(gen_text)
                for item in parsed:
                    s = sanitize_row(
                        item,
                        category_list=categories,
                        polarity_list=polarities,
                        category_set=category_set,
                        polarity_set=polarity_set,
                    )
                    cand_counter[(s["AspectTerms"], s["OpinionTerms"], s["Categories"], s["Polarities"])] += 1

            kept = [quad for quad, cnt in cand_counter.items() if cnt >= vote_threshold]
            if not kept and cand_counter:
                kept = [cand_counter.most_common(1)[0][0]]

            # Drop placeholder row when there are concrete predictions.
            placeholder = ("_", "_", "_", "_")
            kept = [x for x in kept if x != placeholder] or ([(placeholder)] if placeholder in cand_counter or not kept else kept)

            # Reduce obvious noisy rows.
            cleaned: list[tuple[str, str, str, str]] = []
            for a, o, c, p_label in kept:
                if c == "_" or p_label == "_":
                    continue
                if len(a) > 40 or len(o) > 40:
                    continue
                cleaned.append((a, o, c, p_label))
            if not cleaned:
                cleaned = [placeholder]

            for a, o, c, p_label in cleaned:
                rows.append((rid[b], a, o, c, p_label))

        if progress_log_interval > 0 and ((i + len(rid)) % progress_log_interval == 0 or (i + len(rid)) >= total):
            print(f"[Infer] processed {min(i + len(rid), total)}/{total}")

    pred = pd.DataFrame(rows, columns=["id", *QUAD_COLS])
    pred = pred.drop_duplicates(subset=["id", *QUAD_COLS]).sort_values("id").reset_index(drop=True)
    return pred


def get_fold_data(reviews: pd.DataFrame, labels: pd.DataFrame, n_splits: int, kfold_seed: int, fold: int):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=kfold_seed)
    tr_idx, va_idx = list(kf.split(reviews["id"].values))[fold]
    tr_ids = set(reviews.iloc[tr_idx]["id"].tolist())
    va_ids = set(reviews.iloc[va_idx]["id"].tolist())
    tr_reviews = reviews[reviews["id"].isin(tr_ids)].reset_index(drop=True)
    va_reviews = reviews[reviews["id"].isin(va_ids)].reset_index(drop=True)
    tr_labels = labels[labels["id"].isin(tr_ids)].reset_index(drop=True)
    va_labels = labels[labels["id"].isin(va_ids)].reset_index(drop=True)
    return tr_reviews, va_reviews, tr_labels, va_labels


def pick_categories_polarities(labels_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categories = sorted([x for x in labels_df["Categories"].astype(str).unique().tolist() if x != "_"])
    polarities = sorted([x for x in labels_df["Polarities"].astype(str).unique().tolist() if x != "_"])
    return categories, polarities


def run_train_fold(config_path: Path, fold: int, eval_only: bool, resume_from: str | None) -> None:
    normalize_hf_env()
    cfg = load_cfg(config_path)
    set_seed(int(cfg.get("seed", 42)))

    paths = discover_data_paths(ROOT)
    reviews = read_csv_utf8(paths.train_reviews)
    labels = normalize_label_df(read_csv_utf8(paths.train_labels))
    labels = labels.drop_duplicates(subset=["id", *QUAD_COLS]).reset_index(drop=True)

    tr_reviews, va_reviews, tr_labels, va_labels = get_fold_data(
        reviews=reviews,
        labels=labels,
        n_splits=int(cfg["data"]["n_splits"]),
        kfold_seed=int(cfg["data"].get("kfold_seed", cfg.get("seed", 42))),
        fold=fold,
    )
    categories, polarities = pick_categories_polarities(labels)
    category_set = set(categories)
    polarity_set = set(polarities)
    prompt_cfg = cfg.get("prompt", {}) if isinstance(cfg.get("prompt", {}), dict) else {}
    prompt_style = str(prompt_cfg.get("style", "en_v1"))

    model_name = str(cfg["model"]["model_name"])
    env_model_name = os.getenv("MODEL_NAME_OR_PATH", "").strip()
    if env_model_name:
        model_name = env_model_name
    hub_cfg = cfg.get("hub", {}) if isinstance(cfg.get("hub", {}), dict) else {}
    local_raw = hub_cfg.get("local_files_only", False)
    local_files_only = parse_bool(str(local_raw), default=False) if isinstance(local_raw, str) else bool(local_raw)
    env_local_files_only = os.getenv("LOCAL_FILES_ONLY", "").strip()
    if env_local_files_only:
        local_files_only = parse_bool(env_local_files_only, default=local_files_only)
    hf_token = resolve_hf_token(hub_cfg)
    ensure_local_model_exists(model_name=model_name, local_files_only=local_files_only)
    use_fp16, use_bf16 = resolve_precision(cfg.get("train", {}))
    print(
        f"[Runtime] model={model_name} local_files_only={local_files_only} "
        f"token={'set' if hf_token else 'unset'} fp16={use_fp16} bf16={use_bf16}"
    )
    out_dir = ROOT / "outputs" / f"qwen35_lora_fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"

    if not eval_only:
        tr_records = build_records(
            tr_reviews,
            tr_labels,
            categories=categories,
            polarities=polarities,
            prompt_style=prompt_style,
        )
        va_records = build_records(
            va_reviews,
            va_labels,
            categories=categories,
            polarities=polarities,
            prompt_style=prompt_style,
        )
        max_train_samples = int(cfg["data"].get("max_train_samples", -1))
        max_val_samples = int(cfg["data"].get("max_val_samples", -1))
        if max_train_samples > 0:
            tr_records = tr_records[:max_train_samples]
        if max_val_samples > 0:
            va_records = va_records[:max_val_samples]

        model, tokenizer = load_lora_train_model(
            model_name=model_name,
            lora_r=int(cfg["lora"]["r"]),
            lora_alpha=int(cfg["lora"]["alpha"]),
            lora_dropout=float(cfg["lora"]["dropout"]),
            use_bf16=use_bf16,
            local_files_only=local_files_only,
            hf_token=hf_token,
        )
        train_ds = PromptTargetDataset(
            tr_records,
            tokenizer=tokenizer,
            max_length=int(cfg["data"]["max_length"]),
        )
        eval_ds = PromptTargetDataset(
            va_records,
            tokenizer=tokenizer,
            max_length=int(cfg["data"]["max_length"]),
        )
        collator = DataCollatorForCausalLM(tokenizer=tokenizer)

        steps_per_epoch = math.ceil(len(train_ds) / max(1, int(cfg["train"]["batch_size"])))
        print(
            f"[Train] fold={fold} model={model_name} train={len(train_ds)} val={len(eval_ds)} "
            f"steps_per_epoch~{steps_per_epoch}"
        )

        args = build_training_args(
            out_dir=out_dir,
            resume_from=resume_from,
            train_cfg=cfg["train"],
            seed=int(cfg.get("seed", 42)),
            use_fp16=use_fp16,
            use_bf16=use_bf16,
        )
        trainer = build_trainer(
            model=model,
            args=args,
            train_ds=train_ds,
            collator=collator,
            tokenizer=tokenizer,
        )
        train_sig = inspect.signature(trainer.train)
        if "resume_from_checkpoint" in train_sig.parameters:
            trainer.train(resume_from_checkpoint=resume_from)
        else:
            if resume_from:
                print("[WARN] trainer.train does not accept resume_from_checkpoint; resume ignored.")
            trainer.train()
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"[Train] adapter saved to: {adapter_dir}")
    elif not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found for eval_only mode: {adapter_dir}")

    infer_model, infer_tokenizer = load_lora_infer_model(
        model_name=model_name,
        adapter_dir=adapter_dir,
        use_bf16=use_bf16,
        local_files_only=local_files_only,
        hf_token=hf_token,
    )
    pred = predict_reviews(
        model=infer_model,
        tokenizer=infer_tokenizer,
        reviews_df=va_reviews,
        categories=categories,
        polarities=polarities,
        category_set=category_set,
        polarity_set=polarity_set,
        max_prompt_length=int(cfg["inference"].get("max_prompt_length", cfg["data"]["max_length"])),
        max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
        batch_size=int(cfg["inference"]["batch_size"]),
        do_sample=bool(cfg["inference"].get("do_sample", False)),
        temperature=float(cfg["inference"].get("temperature", 0.8)),
        top_p=float(cfg["inference"].get("top_p", 0.9)),
        num_beams=int(cfg["inference"].get("num_beams", 1)),
        num_return_sequences=int(cfg["inference"].get("num_return_sequences", 1)),
        repetition_penalty=float(cfg["inference"].get("repetition_penalty", 1.0)),
        min_votes=int(cfg["inference"].get("min_votes", 1)),
        progress_log_interval=int(cfg["inference"].get("progress_log_interval", 50)),
        prompt_style=prompt_style,
    )
    score = score_quadruples(pred[["id", *QUAD_COLS]], va_labels[["id", *QUAD_COLS]])
    metrics = {
        "fold": fold,
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "pred_count": score.pred_count,
        "gold_count": score.gold_count,
        "hit_count": score.hit_count,
        "adapter_dir": str(adapter_dir),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    report_path = ROOT / "reports" / f"qwen35_lora_fold{fold}_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pred_path = ROOT / "outputs" / f"qwen35_lora_fold{fold}_pred.csv"
    save_csv_no_header(pred[["id", *QUAD_COLS]], pred_path)
    print(f"[Eval] saved metrics: {report_path}")
    print(f"[Eval] saved val pred: {pred_path}")


def run_predict_test(config_path: Path, adapter_dir: Path, output_path: Path) -> None:
    normalize_hf_env()
    cfg = load_cfg(config_path)
    set_seed(int(cfg.get("seed", 42)))

    paths = discover_data_paths(ROOT)
    train_labels = normalize_label_df(read_csv_utf8(paths.train_labels))
    test_reviews = read_csv_utf8(paths.test_reviews)
    categories, polarities = pick_categories_polarities(train_labels)
    category_set = set(categories)
    polarity_set = set(polarities)
    prompt_cfg = cfg.get("prompt", {}) if isinstance(cfg.get("prompt", {}), dict) else {}
    prompt_style = str(prompt_cfg.get("style", "en_v1"))

    model_name = str(cfg["model"]["model_name"])
    env_model_name = os.getenv("MODEL_NAME_OR_PATH", "").strip()
    if env_model_name:
        model_name = env_model_name
    hub_cfg = cfg.get("hub", {}) if isinstance(cfg.get("hub", {}), dict) else {}
    local_raw = hub_cfg.get("local_files_only", False)
    local_files_only = parse_bool(str(local_raw), default=False) if isinstance(local_raw, str) else bool(local_raw)
    env_local_files_only = os.getenv("LOCAL_FILES_ONLY", "").strip()
    if env_local_files_only:
        local_files_only = parse_bool(env_local_files_only, default=local_files_only)
    hf_token = resolve_hf_token(hub_cfg)
    ensure_local_model_exists(model_name=model_name, local_files_only=local_files_only)
    _, use_bf16 = resolve_precision(cfg.get("train", {}))
    print(
        f"[Runtime] model={model_name} local_files_only={local_files_only} "
        f"token={'set' if hf_token else 'unset'} bf16={use_bf16}"
    )
    model, tokenizer = load_lora_infer_model(
        model_name=model_name,
        adapter_dir=adapter_dir,
        use_bf16=use_bf16,
        local_files_only=local_files_only,
        hf_token=hf_token,
    )
    pred = predict_reviews(
        model=model,
        tokenizer=tokenizer,
        reviews_df=test_reviews,
        categories=categories,
        polarities=polarities,
        category_set=category_set,
        polarity_set=polarity_set,
        max_prompt_length=int(cfg["inference"].get("max_prompt_length", cfg["data"]["max_length"])),
        max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
        batch_size=int(cfg["inference"]["batch_size"]),
        do_sample=bool(cfg["inference"].get("do_sample", False)),
        temperature=float(cfg["inference"].get("temperature", 0.8)),
        top_p=float(cfg["inference"].get("top_p", 0.9)),
        num_beams=int(cfg["inference"].get("num_beams", 1)),
        num_return_sequences=int(cfg["inference"].get("num_return_sequences", 1)),
        repetition_penalty=float(cfg["inference"].get("repetition_penalty", 1.0)),
        min_votes=int(cfg["inference"].get("min_votes", 1)),
        progress_log_interval=int(cfg["inference"].get("progress_log_interval", 50)),
        prompt_style=prompt_style,
    )
    save_csv_no_header(pred[["id", *QUAD_COLS]], output_path)
    print(f"[Predict] saved submission: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B LoRA pipeline for Tianchi review ABSA.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train-fold", help="Train one fold and evaluate on its val split.")
    p_train.add_argument("--config", type=Path, default=ROOT / "configs" / "qwen35_9b_lora.yaml")
    p_train.add_argument("--fold", type=int, default=0)
    p_train.add_argument("--eval-only", action="store_true")
    p_train.add_argument("--resume-from", type=str, default=None, help="HF checkpoint dir, e.g. outputs/.../checkpoint-100")

    p_pred = sub.add_parser("predict-test", help="Generate test submission using a trained LoRA adapter.")
    p_pred.add_argument("--config", type=Path, default=ROOT / "configs" / "qwen35_9b_lora.yaml")
    p_pred.add_argument("--adapter-dir", type=Path, required=True)
    p_pred.add_argument("--output", type=Path, default=ROOT / "outputs" / "submit_qwen35_lora.csv")

    args = parser.parse_args()
    if args.cmd == "train-fold":
        run_train_fold(
            config_path=args.config,
            fold=int(args.fold),
            eval_only=bool(args.eval_only),
            resume_from=args.resume_from,
        )
    elif args.cmd == "predict-test":
        run_predict_test(
            config_path=args.config,
            adapter_dir=args.adapter_dir,
            output_path=args.output,
        )
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
