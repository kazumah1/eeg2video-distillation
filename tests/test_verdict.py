"""Behavior tests for decision-table verdicts and montage contrasts on mi_erasure reports."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

N_TEST = 200  # chance top-1 = 0.005
CHANCE = {"category": 0.1}  # 10 coarse categories


def probes(edges=0.6, category=0.5):
    return {"edges": {"r2": edges}, "category": {"balanced_accuracy": category}}


def view(base_top1=0.30, target_top1=0.10, control_top1=0.28,
         target_probes=None, control_probes=None, base_probes=None):
    """One repetition budget of an mi_erasure subject report, erasing 'edges'."""
    control = {"top1": control_top1, "margin": 0.0,
               "probes": control_probes or probes(0.55, 0.48)}
    target = {"top1": target_top1, "margin": 0.0,
              "probes": target_probes or probes(0.0, 0.48)}
    return {
        "baseline": {"top1": base_top1, "margin": 0.0, "probes": base_probes or probes()},
        "interventions": {"edges": {
            "rank": 1, "target": target, "controls": [control, control],
            "target_l2_by_test_index": [1.0] * N_TEST,
            "excess_damage": {"top1": control_top1 - target_top1, "margin": 0.0},
        }},
    }


def report(per_subject_views, budget="80", **meta):
    return {
        "annotations": "a.csv", "descriptors": ["edges"], "category": "category",
        "strength": 1.0, "gallery": "clean", "controls": 2, "ridge": 1e-4,
        **meta,
        "subjects": {str(s): {"views": {budget: v}} for s, v in enumerate(per_subject_views, 1)},
    }


class VerdictTest(unittest.TestCase):
    def test_removed_target_with_large_excess_damage_is_specific_effect(self):
        from eegdistill.verdict import classify

        subjects = [view(target_top1=0.10 + 0.01 * i, control_top1=0.28) for i in range(10)]
        result = classify(report(subjects), "edges", "80", sesoi=0.02, probe_chance=CHANCE)
        self.assertEqual(result["verdict"], "specific_effect")
        self.assertEqual(result["excess_top1"]["n_subjects"], 10)
        self.assertGreater(result["excess_top1"]["ci95"][0], 0.02)

    def test_targeted_and_random_edits_both_at_chance_is_floor_not_null(self):
        from eegdistill.verdict import classify

        subjects = [view(target_top1=0.005, control_top1=0.005 + 0.0005 * (i % 2))
                    for i in range(10)]
        result = classify(report(subjects), "edges", "80", sesoi=0.02, probe_chance=CHANCE)
        self.assertEqual(result["verdict"], "floor")
        self.assertAlmostEqual(result["chance_top1"], 1 / N_TEST)

    def test_baseline_near_chance_is_floor(self):
        from eegdistill.verdict import classify

        subjects = [view(base_top1=0.008, target_top1=0.005, control_top1=0.007)
                    for _ in range(10)]
        result = classify(report(subjects, budget="1"), "edges", "1", sesoi=0.02, probe_chance=CHANCE)
        self.assertEqual(result["verdict"], "floor")

    def test_target_still_recoverable_after_erasure_is_incomplete_removal(self):
        from eegdistill.verdict import classify

        leaky = probes(edges=0.4, category=0.48)  # baseline edges r2 = 0.6
        subjects = [view(target_top1=0.10, control_top1=0.28, target_probes=leaky)
                    for _ in range(10)]
        result = classify(report(subjects), "edges", "80", sesoi=0.02, probe_chance=CHANCE)
        self.assertEqual(result["verdict"], "incomplete_removal")
        self.assertAlmostEqual(result["target_remaining"], 0.4 / 0.6)

    def test_erasure_that_also_removes_category_is_collateral(self):
        from eegdistill.verdict import classify

        wiped = probes(edges=0.0, category=0.2)  # baseline 0.5, chance 0.1
        subjects = [view(target_top1=0.10, control_top1=0.28, target_probes=wiped)
                    for _ in range(10)]
        result = classify(report(subjects), "edges", "80", sesoi=0.02, probe_chance=CHANCE)
        self.assertEqual(result["verdict"], "collateral")
        # Target leaves 25% of above-chance category score; controls leave 95%.
        self.assertAlmostEqual(result["collateral_extra_loss"]["category"], 0.70)

    def test_excess_damage_tightly_near_zero_is_bounded_null_and_wide_is_inconclusive(self):
        from eegdistill.verdict import classify

        tight = [view(target_top1=0.20 + 0.002 * (i % 3 - 1), control_top1=0.20)
                 for i in range(10)]
        wide = [view(target_top1=0.20 + 0.08 * (i % 3 - 1), control_top1=0.20)
                for i in range(10)]
        kwargs = dict(sesoi=0.02, probe_chance=CHANCE)
        self.assertEqual(classify(report(tight), "edges", "80", **kwargs)["verdict"],
                         "bounded_null")
        self.assertEqual(classify(report(wide), "edges", "80", **kwargs)["verdict"],
                         "inconclusive")

    def test_categorical_probe_without_chance_level_is_rejected(self):
        from eegdistill.verdict import classify

        with self.assertRaises(ValueError):
            classify(report([view(), view()]), "edges", "80", sesoi=0.02)


class MontageContrastTest(unittest.TestCase):
    def test_contrast_pairs_subjects_and_differences_excess_damage_directly(self):
        from eegdistill.verdict import montage_contrast

        occ = report([view(base_top1=0.55, target_top1=0.30 - 0.01 * i, control_top1=0.45)
                      for i in range(10)])
        crown = report([view(base_top1=0.17, target_top1=0.12, control_top1=0.17)
                        for _ in range(9)])  # subject 10 missing for Crown
        result = montage_contrast(occ, crown, "edges", "80")
        diff = result["excess_top1_a_minus_b"]
        self.assertEqual(diff["n_subjects"], 9)
        self.assertEqual(result["unpaired_subjects"], ["10"])
        self.assertAlmostEqual(diff["by_subject"]["1"], 0.15 - 0.05)
        self.assertAlmostEqual(diff["by_subject"]["9"], 0.23 - 0.05)
        self.assertGreater(diff["ci95"][0], 0)
        self.assertAlmostEqual(result["baseline_gap_top1"]["mean"], 0.38)

    def test_contrast_rejects_reports_with_different_analysis_settings(self):
        from eegdistill.verdict import montage_contrast

        a = report([view(), view()])
        b = report([view(), view()], strength=0.5)
        with self.assertRaisesRegex(ValueError, "strength"):
            montage_contrast(a, b, "edges", "80")


if __name__ == "__main__":
    unittest.main()
