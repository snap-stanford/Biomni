import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import SimpleITK as sitk
from PIL import Image

REPO_ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def bioimaging_module():
    """Load bioimaging.py without Biomni's nnUNet and PyTorch environment."""
    temporary_modules = {}

    def install_stub(name, module):
        temporary_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    nibabel = ModuleType("nibabel")
    requests = ModuleType("requests")
    install_stub("nibabel", nibabel)
    install_stub("requests", requests)

    torch = ModuleType("torch")
    torch_serialization = ModuleType("torch.serialization")
    torch_serialization.add_safe_globals = lambda _items: None
    torch.serialization = torch_serialization
    install_stub("torch", torch)
    install_stub("torch.serialization", torch_serialization)

    nnunet = ModuleType("nnunet")
    nnunet_inference = ModuleType("nnunet.inference")
    nnunet_predict = ModuleType("nnunet.inference.predict")
    nnunet_predict.predict_from_folder = lambda *_args, **_kwargs: None
    install_stub("nnunet", nnunet)
    install_stub("nnunet.inference", nnunet_inference)
    install_stub("nnunet.inference.predict", nnunet_predict)

    try:
        spec = importlib.util.spec_from_file_location(
            "bioimaging_under_test",
            REPO_ROOT / "biomni/tool/bioimaging.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, previous in temporary_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def make_scalar_image(shape):
    coordinates = np.indices(shape, dtype=np.float32)
    squared_radius = sum((axis - (size - 1) / 2.0) ** 2 for axis, size in zip(coordinates, shape, strict=True))
    image = sitk.GetImageFromArray(np.exp(-squared_radius / 18.0).astype(np.float32))
    image.SetSpacing(tuple(1.0 + 0.1 * index for index in range(image.GetDimension())))
    return image


def write_image(image, path):
    sitk.WriteImage(image, str(path))
    return str(path)


def assert_png(path):
    path = Path(path)
    assert path.is_file()
    assert path.stat().st_size > 0
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width > 100
        assert image.height > 100


def test_standalone_visualization_creates_four_plots_and_metrics(bioimaging_module, tmp_path):
    import matplotlib.pyplot as plt

    fixed = make_scalar_image((7, 20, 18))
    moving_array = np.roll(sitk.GetArrayFromImage(fixed), shift=2, axis=2)
    moving = sitk.GetImageFromArray(moving_array)
    moving.CopyInformation(fixed)
    moving.SetOrigin((1.5, 0.0, 0.0))
    registered = sitk.Image(fixed)

    fixed_path = write_image(fixed, tmp_path / "fixed.mha")
    moving_path = write_image(moving, tmp_path / "moving.mha")
    registered_path = write_image(registered, tmp_path / "registered.mha")
    output_dir = tmp_path / "plots"

    result = bioimaging_module.create_registration_visualization(
        fixed_path,
        moving_path,
        registered_path,
        str(output_dir),
        prefix="case01",
    )

    for key in ("comparison_path", "difference_path", "overlay_path", "metrics_path"):
        assert_png(result[key])
        assert Path(result[key]).parent == output_dir
        assert Path(result[key]).name.startswith("case01_")

    assert result["metrics_after"]["mean_squares"] == pytest.approx(0.0, abs=1e-10)
    assert result["metrics_after"]["correlation"] == pytest.approx(1.0, abs=1e-7)
    assert result["metrics_before"]["mean_squares"] < 0.0
    json.dumps(result)
    assert plt.get_fignums() == []


def test_visualization_supports_2d_images(bioimaging_module, tmp_path):
    fixed = make_scalar_image((24, 20))
    moving = sitk.GetImageFromArray(np.roll(sitk.GetArrayFromImage(fixed), shift=1, axis=1))
    moving.CopyInformation(fixed)
    registered = sitk.Image(fixed)

    result = bioimaging_module.ImageRegistrationTool().create_registration_visualization(
        fixed,
        moving,
        registered,
        str(tmp_path),
        prefix="two_dimensional",
    )

    assert result["metrics_after"]["normalized_correlation"] == pytest.approx(1.0, abs=1e-7)
    assert_png(result["overlay_path"])


@pytest.mark.parametrize("prefix", ["", "  ", ".", "..", "../escape", "folder/name", r"folder\name"])
def test_visualization_rejects_unsafe_prefixes(bioimaging_module, tmp_path, prefix):
    image = make_scalar_image((12, 12))

    with pytest.raises(ValueError, match="prefix"):
        bioimaging_module.ImageRegistrationTool().create_registration_visualization(
            image,
            image,
            image,
            str(tmp_path),
            prefix=prefix,
        )


def test_visualization_rejects_mixed_dimensions(bioimaging_module, tmp_path):
    fixed = make_scalar_image((5, 12, 12))
    moving = make_scalar_image((12, 12))

    with pytest.raises(ValueError, match="same dimension"):
        bioimaging_module.ImageRegistrationTool().create_registration_visualization(
            fixed,
            moving,
            fixed,
            str(tmp_path),
        )


def test_visualization_rejects_vector_images(bioimaging_module, tmp_path):
    scalar = make_scalar_image((12, 12))
    vector = sitk.Compose(scalar, scalar)

    with pytest.raises(ValueError, match="moving_image must be a scalar image"):
        bioimaging_module.ImageRegistrationTool().create_registration_visualization(
            scalar,
            vector,
            scalar,
            str(tmp_path),
        )


@pytest.mark.parametrize(
    ("function_name", "prefix"),
    [
        ("quick_rigid_registration", "rigid"),
        ("quick_affine_registration", "affine"),
        ("quick_deformable_registration", "deformable"),
    ],
)
def test_quick_registration_honors_visualization_flag(
    bioimaging_module,
    monkeypatch,
    tmp_path,
    function_name,
    prefix,
):
    image = make_scalar_image((5, 12, 12))
    visualization_calls = []
    tool_class = bioimaging_module.ImageRegistrationTool

    monkeypatch.setattr(tool_class, "load_image", lambda _self, _path: image)
    monkeypatch.setattr(tool_class, "create_rigid_transform", lambda _self, *_args: "transform")
    monkeypatch.setattr(tool_class, "create_affine_transform", lambda _self, *_args: "transform")
    monkeypatch.setattr(tool_class, "create_deformable_transform", lambda _self, *_args: "transform")
    monkeypatch.setattr(tool_class, "setup_registration_method", lambda _self, *_args: "method")
    monkeypatch.setattr(tool_class, "register_images", lambda _self, *_args: ("final_transform", image))
    monkeypatch.setattr(tool_class, "save_image", lambda _self, *_args: None)
    monkeypatch.setattr(
        tool_class,
        "calculate_similarity_metrics",
        lambda _self, *_args: {
            "mutual_information": 1.0,
            "mean_squares": 0.0,
            "correlation": 1.0,
            "normalized_correlation": 1.0,
        },
    )

    def fake_visualization(_self, *_args, **kwargs):
        visualization_calls.append(kwargs["prefix"])
        return {"comparison_path": f"{kwargs['prefix']}_comparison.png"}

    monkeypatch.setattr(tool_class, "create_registration_visualization", fake_visualization)
    monkeypatch.setattr(bioimaging_module.sitk, "WriteTransform", lambda *_args: None)
    registration_function = getattr(bioimaging_module, function_name)

    result = registration_function(
        "fixed.mha",
        "moving.mha",
        str(tmp_path / "enabled"),
        preprocess=False,
        create_visualizations=True,
    )
    result_without_plots = registration_function(
        "fixed.mha",
        "moving.mha",
        str(tmp_path / "disabled"),
        preprocess=False,
        create_visualizations=False,
    )

    assert result["visualizations"] == {"comparison_path": f"{prefix}_comparison.png"}
    assert "visualizations" not in result_without_plots
    assert visualization_calls == [prefix]


def test_registered_description_has_matching_implementation(bioimaging_module):
    namespace = runpy.run_path(REPO_ROOT / "biomni/tool/tool_description/bioimaging.py")
    schema = next(item for item in namespace["description"] if item["name"] == "create_registration_visualization")

    assert callable(bioimaging_module.create_registration_visualization)
    assert [parameter["name"] for parameter in schema["required_parameters"]] == [
        "fixed_image_path",
        "moving_image_path",
        "registered_image_path",
        "output_dir",
    ]
