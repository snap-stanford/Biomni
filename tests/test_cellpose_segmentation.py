import inspect
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
from biomni.tool.microbiology import segment_cells_with_deep_learning
from biomni.tool.tool_description.microbiology import description
from skimage import io


class FakeCellposeModel:
    instances = []
    eval_result = None

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.eval_calls = []
        self.__class__.instances.append(self)

    def eval(self, image, **kwargs):
        self.eval_calls.append((image, kwargs))
        if self.__class__.eval_result is not None:
            return self.__class__.eval_result

        channel_axis = kwargs["channel_axis"]
        plane_shape = list(image.shape)
        if channel_axis is not None:
            plane_shape.pop(channel_axis % image.ndim)
        masks = np.zeros(plane_shape, dtype=np.int32)
        masks[1:3, 1:4] = 1
        masks[4:6, 5:8] = 3
        return masks, {"flow": "unused"}, np.array([0.25, 0.75])


def fake_cellpose_module():
    models = types.SimpleNamespace(
        CellposeModel=FakeCellposeModel,
        MODEL_NAMES=["cpsam"],
        get_user_models=lambda: ["custom-model"],
    )
    return types.SimpleNamespace(models=models)


class CellposeSegmentationTest(unittest.TestCase):
    def setUp(self):
        FakeCellposeModel.instances = []
        FakeCellposeModel.eval_result = None
        self.cellpose_patch = mock.patch.dict(sys.modules, {"cellpose": fake_cellpose_module()})
        self.cellpose_patch.start()

    def tearDown(self):
        self.cellpose_patch.stop()

    def test_cellpose_4_rgb_segmentation_and_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "rgb.png"
            save_dir = root / "results"
            image = np.zeros((8, 9, 3), dtype=np.uint8)
            image[..., 1] = 100
            io.imsave(image_path, image, check_contrast=False)

            log = segment_cells_with_deep_learning(
                image_path,
                save_dir=save_dir,
                use_gpu=True,
                flow_threshold=0.5,
                cellprob_threshold=-0.25,
                min_size=7,
            )

            self.assertIn("Detected 2 cells", log)
            self.assertIn("native image scale (no rescaling)", log)
            self.assertNotIn("automatically estimated", log)
            model = FakeCellposeModel.instances[0]
            self.assertEqual(model.init_kwargs, {"pretrained_model": "cpsam", "gpu": True})
            evaluated_image, eval_kwargs = model.eval_calls[0]
            np.testing.assert_array_equal(evaluated_image, image)
            self.assertEqual(
                eval_kwargs,
                {
                    "diameter": None,
                    "channel_axis": -1,
                    "flow_threshold": 0.5,
                    "cellprob_threshold": -0.25,
                    "min_size": 7,
                    "do_3D": False,
                },
            )

            mask = io.imread(save_dir / "rgb_masks.tif")
            self.assertEqual(set(np.unique(mask)), {0, 1, 3})
            self.assertEqual(mask.shape, image.shape[:2])
            self.assertTrue((save_dir / "rgb_outlines.png").is_file())

    def test_infers_channel_first_input(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "channel-first.tif"
            image = np.zeros((2, 8, 9), dtype=np.uint8)
            io.imsave(image_path, image, check_contrast=False)

            log = segment_cells_with_deep_learning(image_path, save_dir=root / "results")

            self.assertIn("Channel axis: 0", log)
            _, eval_kwargs = FakeCellposeModel.instances[0].eval_calls[0]
            self.assertEqual(eval_kwargs["channel_axis"], 0)

    def test_rejects_unknown_model_instead_of_silently_falling_back(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "gray.tif"
            io.imsave(image_path, np.zeros((8, 9), dtype=np.uint8), check_contrast=False)

            log = segment_cells_with_deep_learning(
                image_path,
                model_type="bact_fluor_omni",
                save_dir=root / "results",
            )

            self.assertIn("unknown model 'bact_fluor_omni'", log)
            self.assertIn("cpsam", log)
            self.assertEqual(FakeCellposeModel.instances, [])

    def test_rejects_ambiguous_volume_and_channel_axis_on_grayscale(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            volume_path = root / "volume.tif"
            gray_path = root / "gray.tif"
            io.imsave(volume_path, np.zeros((5, 6, 7), dtype=np.uint8), check_contrast=False)
            io.imsave(gray_path, np.zeros((8, 9), dtype=np.uint8), check_contrast=False)

            volume_log = segment_cells_with_deep_learning(volume_path, save_dir=root / "volume-results")
            gray_log = segment_cells_with_deep_learning(
                gray_path,
                channel_axis=0,
                save_dir=root / "gray-results",
            )

            self.assertIn("cannot infer a channel axis", volume_log)
            self.assertIn("channel_axis must be None", gray_log)
            self.assertEqual(FakeCellposeModel.instances, [])

    def test_rejects_non_v4_eval_result(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "gray.tif"
            io.imsave(image_path, np.zeros((8, 9), dtype=np.uint8), check_contrast=False)
            FakeCellposeModel.eval_result = (np.zeros((8, 9)), {"flow": "unused"})

            log = segment_cells_with_deep_learning(image_path, save_dir=root / "results")

            self.assertIn("Cellpose 4 model.eval() must return 3 values, got 2", log)

    def test_tool_description_matches_function_signature(self):
        tool = next(item for item in description if item["name"] == "segment_cells_with_deep_learning")
        schema_defaults = {item["name"]: item["default"] for item in tool["optional_parameters"]}
        function_defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(segment_cells_with_deep_learning).parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }

        self.assertEqual(schema_defaults, function_defaults)
        self.assertEqual(schema_defaults["model_type"], "cpsam")


if __name__ == "__main__":
    unittest.main()
