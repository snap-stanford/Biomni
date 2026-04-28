#!/usr/bin/env python3

"""
CAi backend tool: libinvent / default

这个脚本是 CAi 后端工具系统里 libinvent 工具的标准入口。
它负责：

1. 从当前 job 工作目录读取 params.json
2. 对输入 scaffold 做规范化与校验
3. 调用真实 Lib-INVENT 执行 decoration
4. 做去重、exclude_smiles 过滤、必要的补采样
5. 把结果写入 result.json

说明：
- 成功结果保持当前结构，不做大的改动
- 失败结果改成“极简错误协议”，方便大模型直接消费
"""

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

# ============================================================
# 1. 基础路径与依赖初始化
# ============================================================

TOOL_ROOT = Path(__file__).resolve().parent
LIBINVENT_ROOT = TOOL_ROOT / "Lib-INVENT"

if not LIBINVENT_ROOT.exists():
    raise FileNotFoundError(f"Lib-INVENT directory not found: {LIBINVENT_ROOT}")

if str(LIBINVENT_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBINVENT_ROOT))

from running_modes.manager import Manager

try:
    from rdkit import Chem
except Exception:
    Chem = None


# ============================================================
# 2. 路径展示策略：默认少暴露绝对路径
# ============================================================


def compact_path(
    path: Path | None,
    *,
    include_debug_paths: bool,
    job_dir: Path | None = None,
) -> str | None:
    """
    把路径压缩成更适合返回给 Agent / 上层系统的形式。

    规则：
    - 若 include_debug_paths=True：返回绝对路径
    - 否则：
      1) 若在当前 job_dir 下，返回相对 job_dir 的相对路径
      2) 否则只返回 basename
    """
    if path is None:
        return None

    path = Path(path)

    if include_debug_paths:
        return str(path.resolve())

    if job_dir is not None:
        try:
            return str(path.resolve().relative_to(job_dir.resolve()))
        except Exception:
            pass

    return path.name


# ============================================================
# 3. 极简错误构造
# ============================================================


def build_error_result(
    message: str,
    *,
    error_type: str,
    recoverable: bool,
    validated_input: dict[str, Any] | None = None,
    debug_context: dict[str, Any] | None = None,
    repair_hints: list[str] | None = None,
    suggested_next_actions: list[str] | None = None,
    input_smiles: str | None = None,
    normalized_smiles: str | None = None,
    repair_hint: str | None = None,
    include_debug_paths: bool = False,
) -> dict[str, Any]:
    """
    兼容版失败结果构造函数。

    当前兼容策略：
    1. 继续保留结构化字段：error_type / recoverable / repair_hint / normalized_smiles
    2. 同时把核心修复线索压进顶层 error 字符串
    3. 这样即使 template_tools.py 只透传 error，大模型也还能看到修复提示
    """

    final_repair_hint = repair_hint
    if not final_repair_hint and repair_hints:
        final_repair_hint = repair_hints[0]

    # 把关键修复信息压进 error，兼容当前旧的 template_tools._call_worker_api() 逻辑
    compat_parts = [
        f"error_type={error_type}",
        f"recoverable={str(recoverable).lower()}",
    ]

    if final_repair_hint:
        compat_parts.append(f"repair_hint={final_repair_hint}")

    if normalized_smiles:
        compat_parts.append(f"normalized_smiles={normalized_smiles}")

    compat_error = f"{message} | " + " | ".join(compat_parts)

    result: dict[str, Any] = {
        "success": False,
        "error": compat_error,
        "error_type": error_type,
        "recoverable": recoverable,
        "message": message,
    }

    if input_smiles:
        result["input_smiles"] = input_smiles

    if normalized_smiles:
        result["normalized_smiles"] = normalized_smiles

    if final_repair_hint:
        result["repair_hint"] = final_repair_hint

    if include_debug_paths and debug_context:
        result["debug"] = debug_context

    return result


# ============================================================
# 4. scaffold 规范化与语法校验
# ============================================================


def normalize_scaffold_smiles(smiles: str) -> str:
    """
    对输入 scaffold 做最小限度的确定性修复。

    当前处理：
    - 去空格
    - 把 (*) 转为 ([*])
    - 把部分裸 * 尝试规范成 [*]
    """
    smiles = smiles.strip()

    smiles = smiles.replace("(*)", "([*])")

    # 一些常见裸 * 位置的保守修复
    smiles = re.sub(r"(?<=\))\*", "[*]", smiles)
    smiles = re.sub(r"(?<=O)\*", "[*]", smiles)
    smiles = re.sub(r"(?<=N)\*", "[*]", smiles)
    smiles = re.sub(r"(?<=C)\*", "[*]", smiles)

    return smiles


def rdkit_validate_scaffold(smiles: str) -> tuple[bool, str | None]:
    """
    用 RDKit 对 scaffold 做语法级解析检查。

    返回：
    - (True, None): 通过
    - (False, message): 失败及原因
    """
    if Chem is None:
        return True, None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, f"RDKit cannot parse scaffold SMILES: {smiles}"
        return True, None
    except Exception as e:
        return False, f"RDKit failed to parse scaffold SMILES: {smiles}; error={e}"


def validate_input(params: dict[str, Any]) -> dict[str, Any]:
    """
    校验并规范化输入参数。
    """
    original_smiles = params.get("smiles", "")
    num_decorations = params.get("number_of_decorations_per_scaffold", 32)
    exclude_smiles = params.get("exclude_smiles", [])
    batch_size = params.get("batch_size", 1)
    randomize = params.get("randomize", True)
    run_type = params.get("run_type", "scaffold_decorating")
    model_path = params.get("model_path", str(LIBINVENT_ROOT / "trained_models" / "reaction_based.model"))
    max_rounds = params.get("max_rounds", 5)
    oversample_factor = params.get("oversample_factor", 3)
    max_candidates_per_round = params.get("max_candidates_per_round", 128)
    preview_limit = params.get("preview_limit", 10)
    include_debug_paths = params.get("include_debug_paths", False)

    if not isinstance(original_smiles, str) or not original_smiles.strip():
        raise ValueError("Missing required field: smiles")

    normalized_smiles = normalize_scaffold_smiles(original_smiles)

    if "*" not in normalized_smiles and "[*]" not in normalized_smiles and "[*:" not in normalized_smiles:
        raise ValueError(
            "The input scaffold SMILES must contain at least one attachment point, such as '[*]' or '[*:1]'."
        )

    ok, rdkit_error = rdkit_validate_scaffold(normalized_smiles)
    if not ok:
        raise ValueError(rdkit_error)

    if not isinstance(num_decorations, int) or num_decorations <= 0:
        raise ValueError("number_of_decorations_per_scaffold must be a positive integer")

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    if exclude_smiles is None:
        exclude_smiles = []
    if not isinstance(exclude_smiles, list):
        raise ValueError("exclude_smiles must be a list of SMILES strings")

    exclude_smiles = [x.strip() for x in exclude_smiles if isinstance(x, str) and x.strip()]

    if not isinstance(max_rounds, int) or max_rounds <= 0:
        raise ValueError("max_rounds must be a positive integer")

    if not isinstance(oversample_factor, int) or oversample_factor <= 0:
        raise ValueError("oversample_factor must be a positive integer")

    if not isinstance(max_candidates_per_round, int) or max_candidates_per_round <= 0:
        raise ValueError("max_candidates_per_round must be a positive integer")

    if not isinstance(preview_limit, int) or preview_limit <= 0:
        raise ValueError("preview_limit must be a positive integer")

    return {
        "input_smiles": original_smiles,
        "smiles": normalized_smiles,
        "number_of_decorations_per_scaffold": num_decorations,
        "exclude_smiles": exclude_smiles,
        "batch_size": batch_size,
        "randomize": bool(randomize),
        "run_type": run_type,
        "model_path": model_path,
        "max_rounds": max_rounds,
        "oversample_factor": oversample_factor,
        "max_candidates_per_round": max_candidates_per_round,
        "preview_limit": preview_limit,
        "include_debug_paths": bool(include_debug_paths),
    }


# ============================================================
# 5. Lib-INVENT 配置与输入文件写入
# ============================================================


def write_scaffold_file(scaffold_smiles: str, path: Path) -> None:
    """
    把当前轮次的 scaffold 写入输入文件。
    当前格式：一行一个 scaffold SMILES。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(scaffold_smiles + "\n")


def build_libinvent_config(
    validated: dict[str, Any],
    *,
    round_index: int,
    request_count: int,
) -> dict[str, Any]:
    """
    构造单轮 Lib-INVENT config。
    """
    job_dir = Path.cwd()

    input_scaffold_path = job_dir / f"input_scaffold_round{round_index}.smi"
    output_path = job_dir / f"decorate_output_round{round_index}.csv"
    logging_path = job_dir / f"decorate_log_round{round_index}"

    write_scaffold_file(validated["smiles"], input_scaffold_path)

    return {
        "run_type": validated["run_type"],
        "parameters": {
            "model_path": validated["model_path"],
            "input_scaffold_path": str(input_scaffold_path),
            "output_path": str(output_path),
            "logging_path": str(logging_path),
            "batch_size": validated["batch_size"],
            "number_of_decorations_per_scaffold": request_count,
            "randomize": validated["randomize"],
        },
    }


def validate_paths(config: dict[str, Any]) -> None:
    """
    校验本轮运行所需路径。
    """
    params = config["parameters"]

    model_path = Path(params["model_path"])
    input_scaffold_path = Path(params["input_scaffold_path"])
    output_path = Path(params["output_path"])
    logging_path = Path(params["logging_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not input_scaffold_path.exists():
        raise FileNotFoundError(f"Input scaffold file not found: {input_scaffold_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging_path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 6. CSV 读取、字段识别、结果标准化
# ============================================================


def read_csv_rows(output_path: Path) -> list[dict[str, Any]]:
    """
    读取单轮输出 CSV。
    """
    if not output_path.exists():
        raise FileNotFoundError(f"Expected output file was not created: {output_path}")

    with open(output_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def infer_smiles_key(rows: list[dict[str, Any]]) -> str:
    """
    推断输出表里“生成分子 SMILES”所在列名。
    """
    if not rows:
        return "SMILES"

    candidate_keys = [
        "SMILES",
        "smiles",
        "Smiles",
        "sampled_molecules",
        "Sampled_Molecules",
        "decorated_smiles",
        "generated_smiles",
    ]

    first = rows[0]
    for key in candidate_keys:
        if key in first:
            return key

    for key in first.keys():
        if "smile" in key.lower():
            return key

    return "SMILES"


def normalize_records_from_rows(
    rows: list[dict[str, Any]],
    *,
    input_scaffold: str,
    requested_num: int,
    exclude_smiles: list[str],
    seen_smiles: set[str],
) -> list[dict[str, Any]]:
    """
    将单轮输出 CSV 规范化为统一 records。
    """
    exclude_set = set(exclude_smiles)
    records: list[dict[str, Any]] = []
    smiles_key = infer_smiles_key(rows)

    for row in rows:
        raw_smiles = str(row.get(smiles_key, "")).strip()
        if not raw_smiles:
            continue
        if raw_smiles in exclude_set:
            continue
        if raw_smiles in seen_smiles:
            continue

        seen_smiles.add(raw_smiles)

        records.append(
            {
                "input_scaffold": input_scaffold,
                "SMILES": raw_smiles,
                "generated_smiles": raw_smiles,
                "status": "generated",
                "source": "libinvent",
                "requested_num_decorations": requested_num,
                "excluded_count": len(exclude_smiles),
                "is_duplicate_with_exclude": raw_smiles in exclude_set,
                "raw_row": row,
                "message": "Generated by Lib-INVENT.",
            }
        )

    return records


# ============================================================
# 7. 多轮补采样
# ============================================================


def compute_request_count(
    remaining_needed: int,
    *,
    oversample_factor: int,
    max_candidates_per_round: int,
) -> int:
    """
    计算本轮应该请求多少个候选。
    """
    requested = max(1, remaining_needed * oversample_factor)
    requested = min(requested, max_candidates_per_round)
    return requested


def run_single_round(
    validated: dict[str, Any],
    *,
    round_index: int,
    request_count: int,
) -> tuple[dict[str, Any], Path, Path]:
    """
    执行一轮真实 Lib-INVENT 生成。
    """
    config = build_libinvent_config(
        validated,
        round_index=round_index,
        request_count=request_count,
    )
    validate_paths(config)

    manager = Manager(config)
    manager.run()

    output_path = Path(config["parameters"]["output_path"])
    logging_path = Path(config["parameters"]["logging_path"])
    return config, output_path, logging_path


def collect_records_with_resampling(validated: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    多轮补采样主逻辑。

    目标：
    - 尽量收集到 target_n 个唯一且未被 exclude_smiles 排除的分子
    - 若模型本身给不出这么多，则返回实际拿到的数量
    """
    target_n = validated["number_of_decorations_per_scaffold"]
    exclude_smiles = validated["exclude_smiles"]

    all_records: list[dict[str, Any]] = []
    seen_smiles: set[str] = set()

    round_summaries: list[dict[str, Any]] = []
    latest_config: dict[str, Any] | None = None
    latest_output_path: Path | None = None
    latest_logging_path: Path | None = None

    for round_index in range(1, validated["max_rounds"] + 1):
        remaining_needed = target_n - len(all_records)
        if remaining_needed <= 0:
            break

        request_count = compute_request_count(
            remaining_needed,
            oversample_factor=validated["oversample_factor"],
            max_candidates_per_round=validated["max_candidates_per_round"],
        )

        config, output_path, logging_path = run_single_round(
            validated,
            round_index=round_index,
            request_count=request_count,
        )

        latest_config = config
        latest_output_path = output_path
        latest_logging_path = logging_path

        rows = read_csv_rows(output_path)

        new_records = normalize_records_from_rows(
            rows,
            input_scaffold=validated["smiles"],
            requested_num=target_n,
            exclude_smiles=exclude_smiles,
            seen_smiles=seen_smiles,
        )

        all_records.extend(new_records)

        round_summaries.append(
            {
                "round_index": round_index,
                "requested_candidates": request_count,
                "raw_csv_rows": len(rows),
                "new_unique_records_kept": len(new_records),
                "total_collected_so_far": len(all_records),
                "output_file": compact_path(
                    output_path,
                    include_debug_paths=validated["include_debug_paths"],
                    job_dir=Path.cwd(),
                ),
                "logging_dir": compact_path(
                    logging_path,
                    include_debug_paths=validated["include_debug_paths"],
                    job_dir=Path.cwd(),
                ),
            }
        )

    meta: dict[str, Any] = {
        "target_n": target_n,
        "actual_n_before_truncation": len(all_records),
        "round_summaries": round_summaries,
        "max_rounds": validated["max_rounds"],
        "oversample_factor": validated["oversample_factor"],
        "max_candidates_per_round": validated["max_candidates_per_round"],
    }

    if validated["include_debug_paths"]:
        meta["latest_output_path"] = str(latest_output_path.resolve()) if latest_output_path else None
        meta["latest_logging_path"] = str(latest_logging_path.resolve()) if latest_logging_path else None
        meta["latest_config"] = latest_config
        meta["tool_root"] = str(TOOL_ROOT.resolve())
        meta["libinvent_root"] = str(LIBINVENT_ROOT.resolve())

    return all_records[:target_n], meta


# ============================================================
# 8. 成功结果构造（保持原逻辑）
# ============================================================


def build_success_result(
    records: list[dict[str, Any]],
    *,
    validated: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """
    构造统一成功结果。

    注意：
    - summary 给 Agent / UI 快速读
    - results 保留详细结果
    - outputs 默认不暴露绝对路径
    """
    preview_limit = validated["preview_limit"]
    preview_columns = ["SMILES", "status", "message"]

    preview = [{"SMILES": r["SMILES"], "status": r["status"], "message": r["message"]} for r in records[:preview_limit]]

    actual_n = len(records)
    target_n = validated["number_of_decorations_per_scaffold"]

    partial_message = None
    if actual_n < target_n:
        partial_message = (
            f"Requested {target_n} unique molecules, but only obtained {actual_n} after deduplication and filtering."
        )

    outputs = None
    if meta.get("round_summaries"):
        last_round = meta["round_summaries"][-1]
        outputs = {
            "job_relative_dir": ".",
            "output_file": last_round.get("output_file"),
            "logging_dir": last_round.get("logging_dir"),
        }

    parameters: dict[str, Any] = {
        "batch_size": validated["batch_size"],
        "number_of_decorations_per_scaffold": validated["number_of_decorations_per_scaffold"],
        "randomize": validated["randomize"],
        "exclude_smiles_count": len(validated["exclude_smiles"]),
        "max_rounds": validated["max_rounds"],
        "oversample_factor": validated["oversample_factor"],
        "max_candidates_per_round": validated["max_candidates_per_round"],
    }

    if validated["include_debug_paths"]:
        parameters["model_path"] = str(Path(validated["model_path"]).resolve())

    return {
        "success": True,
        "summary": {
            "total": actual_n,
            "successful": actual_n,
            "failed": 0,
            "row_count": actual_n,
            "columns": preview_columns,
            "preview": preview,
            "partial_message": partial_message,
        },
        "results": records,
        "errors": None,
        "outputs": outputs,
        "parameters": parameters,
        "run_type": validated["run_type"],
        "meta": meta,
    }


# ============================================================
# 9. 主流程（仅简化失败返回）
# ============================================================


def main() -> None:
    """
    主入口流程：

    1. 读取 params.json
    2. 规范化 + 校验输入
    3. 多轮调用 Lib-INVENT 做补采样
    4. 若无结果，则返回极简失败协议
    5. 最终把 result.json 写到当前 job 目录
    """
    validated: dict[str, Any] | None = None
    raw_params: dict[str, Any] | None = None

    try:
        with open("params.json", encoding="utf-8") as f:
            raw_params = json.load(f)

        validated = validate_input(raw_params)
        records, meta = collect_records_with_resampling(validated)

        if not records:
            result = build_error_result(
                "Lib-INVENT completed, but no valid molecules were obtained.",
                error_type="no_valid_molecules_generated",
                recoverable=True,
                input_smiles=validated["input_smiles"],
                normalized_smiles=validated["smiles"],
                repair_hint="Try a simpler scaffold or move the attachment point, then retry.",
                include_debug_paths=validated["include_debug_paths"],
                debug_context={
                    "job_dir": str(Path.cwd().resolve()),
                    "round_count": len(meta.get("round_summaries", [])),
                },
            )
        else:
            result = build_success_result(
                records,
                validated=validated,
                meta=meta,
            )

    except ValueError as e:
        input_smiles = raw_params.get("smiles") if isinstance(raw_params, dict) else None
        normalized_smiles = None
        if isinstance(input_smiles, str) and input_smiles.strip():
            try:
                normalized_smiles = normalize_scaffold_smiles(input_smiles)
            except Exception:
                normalized_smiles = input_smiles

        msg = str(e)
        error_type = "invalid_input"
        repair_hint = "Check the input format and retry."

        if "attachment point" in msg.lower():
            error_type = "missing_attachment_point"
            repair_hint = "Add one attachment point such as '[*]' or '[*:1]' and retry."

        elif (
            "rdkit cannot parse scaffold smiles" in msg.lower()
            or "rdkit failed to parse scaffold smiles" in msg.lower()
        ):
            error_type = "invalid_scaffold_smiles"
            repair_hint = "Try a more RDKit-parseable scaffold while preserving one attachment point."

        elif "batch_size" in msg.lower():
            error_type = "invalid_batch_size"
            repair_hint = "Use a positive integer batch_size, e.g. 1."

        result = build_error_result(
            msg,
            error_type=error_type,
            recoverable=True,
            input_smiles=input_smiles,
            normalized_smiles=normalized_smiles,
            repair_hint=repair_hint,
            include_debug_paths=bool(raw_params.get("include_debug_paths", False))
            if isinstance(raw_params, dict)
            else False,
            debug_context={
                "job_dir": str(Path.cwd().resolve()),
            },
        )

    except FileNotFoundError as e:
        result = build_error_result(
            str(e),
            error_type="missing_required_file",
            recoverable=False,
            input_smiles=raw_params.get("smiles") if isinstance(raw_params, dict) else None,
            normalized_smiles=validated.get("smiles") if isinstance(validated, dict) else None,
            repair_hint=None,
            include_debug_paths=bool(raw_params.get("include_debug_paths", False))
            if isinstance(raw_params, dict)
            else False,
            debug_context={
                "job_dir": str(Path.cwd().resolve()),
                "tool_root": str(TOOL_ROOT.resolve()),
                "libinvent_root": str(LIBINVENT_ROOT.resolve()),
            },
        )

    except Exception as e:
        result = build_error_result(
            str(e),
            error_type="internal_tool_error",
            recoverable=False,
            input_smiles=raw_params.get("smiles") if isinstance(raw_params, dict) else None,
            normalized_smiles=validated.get("smiles") if isinstance(validated, dict) else None,
            repair_hint=None,
            include_debug_paths=bool(raw_params.get("include_debug_paths", False))
            if isinstance(raw_params, dict)
            else False,
            debug_context={
                "job_dir": str(Path.cwd().resolve()),
            },
        )

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
