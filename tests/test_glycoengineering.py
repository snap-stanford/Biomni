import importlib.util
import re
import types
import unittest
from pathlib import Path


def load_glyco_module() -> types.ModuleType:
    """Load biomni/tool/glycoengineering.py without importing the biomni package.
    This avoids package-level imports (e.g., pandas) during test discovery.
    """
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "biomni" / "tool" / "glycoengineering.py"
    spec = importlib.util.spec_from_file_location("biomni_tool_glycoengineering", str(mod_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestGlycoengineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_glyco_module()

    def test_find_n_glycosylation_motifs_basic_and_overlap(self):
        f = self.mod.find_n_glycosylation_motifs

        # Single canonical sequon
        out = f("NVT")
        self.assertIn("Total sequons found: 1", out)
        self.assertRegex(out, r"-\s+1:\s+NVT")

        # X cannot be Proline
        out = f("NPT")
        self.assertIn("Total sequons found: 0", out)

        # Overlapping sequons: NNS (pos1) and NST (pos2)
        seq = "NNST"
        out_no_overlap = f(seq, allow_overlap=False)
        self.assertIn("Total sequons found: 1", out_no_overlap)
        self.assertRegex(out_no_overlap, r"-\s+1:\s+NNS")
        self.assertNotRegex(out_no_overlap, r"-\s+2:\s+NST")

        out_overlap = f(seq, allow_overlap=True)
        self.assertIn("Total sequons found: 2", out_overlap)
        self.assertRegex(out_overlap, r"-\s+1:\s+NNS")
        self.assertRegex(out_overlap, r"-\s+2:\s+NST")

    def test_predict_o_glycosylation_hotspots_density_and_proline_rule(self):
        g = self.mod.predict_o_glycosylation_hotspots

        # S/T rich segment should yield candidates
        out = g("AAAASTTAAAA", window=7, min_st_fraction=0.3)
        self.assertRegex(out, r"Total candidate sites: \d+")
        # Should list at least one 'pos N (' line
        self.assertRegex(out, r"-\s+pos\s+\d+\s+\([ST]\):")

        # Disallow S/T immediately followed by Proline
        out2 = g("A\nSP\nASTPA\n".replace("\n", ""), window=3, min_st_fraction=0.34, disallow_proline_next=True)
        # Ensure no listing for positions where S/T is followed by P
        # The log lists lines like: - pos N (S): ...
        lines = [ln.strip() for ln in out2.splitlines() if ln.strip().startswith("- pos ")]
        for ln in lines:
            m = re.search(r"- pos (\d+) \(([ST])\):", ln)
            if not m:
                continue
            pos = int(m.group(1))
            m.group(2)
            seq = "ASP ASTPA".replace(" ", "")
            if pos < len(seq) and seq[pos] == "P":
                self.fail("Proline-followed S/T site should have been excluded: " + ln)

        # Allowing proline-next should include such positions if density passes
        out3 = g("ASTPA", window=3, min_st_fraction=0.34, disallow_proline_next=False)
        self.assertRegex(out3, r"Total candidate sites: \d+")

    def test_list_glycoengineering_resources_contains_expected_links(self):
        lister = self.mod.list_glycoengineering_resources
        out = lister()
        # Check presence of the URLs referenced by the issue
        self.assertIn("https://gitlab.mpcdf.mpg.de/dioscuri-biophysics/glycoshield-md/", out)
        self.assertIn("https://github.com/CopenhagenCenterForGlycomics", out)
        self.assertIn("https://services.healthtech.dtu.dk/services/NetOGlyc-4.0/", out)


if __name__ == "__main__":
    unittest.main()
