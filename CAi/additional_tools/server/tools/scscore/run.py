import sys
import json
from pathlib import Path
from io import StringIO

# 全局模型缓存
MODEL_CACHE = {}


def interpret_scscore(score: float):
    if score < 1.5:
        return "very easy synthesis"
    elif score < 2.5:
        return "easy synthesis"
    elif score < 3.5:
        return "moderate synthesis difficulty"
    elif score < 4.5:
        return "difficult synthesis"
    else:
        return "very difficult synthesis"


def load_model(model_type="1024bool"):
    """
    Load SCScore model (with cache)
    """

    if model_type in MODEL_CACHE:
        return MODEL_CACHE[model_type]

    script_dir = Path(__file__).resolve().parent
    scscore_path = script_dir / "scscore"

    if str(scscore_path) not in sys.path:
        sys.path.insert(0, str(scscore_path))

    from scscore.standalone_model_numpy import SCScorer

    model_dir = scscore_path / "models"

    model_paths = {
        "1024bool": model_dir / "full_reaxys_model_1024bool" / "model.ckpt-10654.as_numpy.json.gz",
        "2048bool": model_dir / "full_reaxys_model_2048bool" / "model.ckpt-10654.as_numpy.json.gz",
        "1024uint8": model_dir / "full_reaxys_model_1024uint8" / "model.ckpt-10654.as_numpy.json.gz",
    }

    if model_type not in model_paths:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_path = model_paths[model_type]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = SCScorer()

    # suppress model restore output
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        if model_type == "2048bool":
            model.restore(str(model_path), FP_len=2048)
        else:
            model.restore(str(model_path))
    finally:
        sys.stdout = old_stdout

    MODEL_CACHE[model_type] = model

    print(f"SCScore model loaded: {model_type}", file=sys.stderr)

    return model


def calculate_scscore(smiles_list, model_type="1024bool"):

    model = load_model(model_type)

    results = []
    errors = []
    scores = []

    for i, smiles in enumerate(smiles_list):

        try:

            canonical_smiles, score = model.get_score_from_smi(smiles)

            if hasattr(score, "item"):
                score = score.item()
            else:
                score = float(score)

            if canonical_smiles:

                results.append(
                    {
                        "index": i,
                        "input_smiles": smiles,
                        "canonical_smiles": canonical_smiles,
                        "scscore": score,
                        "interpretation": interpret_scscore(score),
                    }
                )

                scores.append(score)

            else:

                errors.append(
                    {
                        "index": i,
                        "smiles": smiles,
                        "error": "invalid smiles",
                    }
                )

        except Exception as e:

            errors.append(
                {
                    "index": i,
                    "smiles": smiles,
                    "error": str(e),
                }
            )

    summary = {
        "total": len(smiles_list),
        "successful": len(results),
        "failed": len(errors),
        "model": model_type,
    }

    if scores:

        scores_sorted = sorted(scores)

        summary.update(
            {
                "avg_scscore": sum(scores) / len(scores),
                "min_scscore": min(scores),
                "max_scscore": max(scores),
                "median_scscore": scores_sorted[len(scores_sorted) // 2],
            }
        )

    return {
        "success": True,
        "summary": summary,
        "results": results,
        "errors": errors if errors else None,
    }


def main():
    try:
        params_file = Path("params.json")
        if not params_file.exists():
            raise ValueError("params.json not found")

        params = json.load(open(params_file))

        smiles = params.get("smiles")
        smiles_list = params.get("smiles_list")
        model_type = params.get("model_type", "1024bool")

        if smiles:
            smiles_list = [smiles]

        if not smiles_list:
            raise ValueError("smiles or smiles_list must be provided")

        result = calculate_scscore(smiles_list, model_type)

    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
        }

    # 将结构化数据写入 result.json
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    # stdout 现在可以安全地用来打印调试信息了
    print("Task completed. Output saved to result.json.")

if __name__ == "__main__":
    main()
