"""conftest: 本地开发环境兼容层。

上游 CI 环境依赖完整，本文件在正常环境不产生任何影响。
仅当本地 numpy 损坏（缺 ndarray）或 biopython 未安装时，
提供最小 stub 让 database.py 可导入（测试只验证纯逻辑）。

判断逻辑：只有 import 真的失败/损坏时才 stub。
"""

import sys
import types

# --- numpy/pandas stub（仅当损坏/缺失时） ---
try:
    import numpy

    has_ndarray = hasattr(numpy, "ndarray")
except Exception:
    has_ndarray = False

if not has_ndarray:
    fake_np = types.ModuleType("numpy")
    fake_np.ndarray = type("ndarray", (), {})
    fake_np.__version__ = "99.0"
    fake_np.array = lambda *a, **k: None
    sys.modules["numpy"] = fake_np

    fake_pd = types.ModuleType("pandas")
    fake_pd.DataFrame = type("DataFrame", (), {})
    sys.modules["pandas"] = fake_pd

# --- Bio (biopython) stub（仅当未安装时） ---
try:
    import Bio  # noqa: F401
except ImportError:
    fake_bio = types.ModuleType("Bio")

    fake_blast = types.ModuleType("Bio.Blast")
    fake_blast.NCBIWWW = types.SimpleNamespace()
    fake_blast.NCBIXML = types.SimpleNamespace()
    fake_blast.__path__ = []
    sys.modules["Bio.Blast"] = fake_blast

    fake_seq = types.ModuleType("Bio.Seq")
    fake_seq.Seq = type("Seq", (), {})
    sys.modules["Bio.Seq"] = fake_seq

    fake_bio.Blast = fake_blast
    fake_bio.Seq = fake_seq
    fake_bio.__path__ = []
    sys.modules["Bio"] = fake_bio
