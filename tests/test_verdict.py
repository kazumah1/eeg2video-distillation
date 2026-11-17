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

    def test_derived_target_counts_its_parent_as_related_not_collateral(self):
        from eegdistill.verdict import classify

        def resid_view():
            v = view(target_top1=0.10, control_top1=0.28)
            for scores in [v["baseline"], v["interventions"]["edges"]["target"],
                           *v["interventions"]["edges"]["controls"]]:
                scores["probes"]["edges_resid"] = {"r2": scores["probes"]["edges"]["r2"]}
            # Residualized edit wipes most raw edge signal; category untouched.
            v["interventions"]["edges"]["target"]["probes"]["edges"] = {"r2": 0.1}
            v["interventions"]["edges_resid"] = v["interventions"].pop("edges")
            return v

        r = report([resid_view() for _ in range(10)],
                   derived={"edges_resid": {"kind": "residualized", "from": "edges",
                                            "on": "category"}})
        result = classify(r, "edges_resid", "80", sesoi=0.02, probe_chance=CHANCE)
        self.assertEqual(result["verdict"], "specific_effect")
        self.assertEqual(set(result["collateral_extra_loss"]), {"category"})
        self.assertAlmostEqual(result["related_remaining"]["edges"], 0.1 / 0.6)

    def test_shuffled_copies_replace_random_edits_as_the_specificity_baseline(self):
        from eegdistill.verdict import classify

        def with_shuffles(i, shuffled_target):
            v = view(target_top1=0.10, control_top1=0.28)  # D vs random = 0.18
            for k in range(2):
                s = view(target_top1=shuffled_target + 0.001 * (i % 3), control_top1=0.28)
                v["interventions"][f"edges_shuf{k}"] = s["interventions"]["edges"]
            return v

        derived = {f"edges_shuf{k}": {"kind": "shuffled", "from": "edges"} for k in range(2)}
        generic = report([with_shuffles(i, 0.105) for i in range(10)], derived=derived)
        result = classify(generic, "edges", "80", sesoi=0.02, probe_chance=CHANCE)
        self.assertGreater(result["excess_top1"]["mean"], 0.17)  # still vs random
        self.assertAlmostEqual(result["excess_vs_shuffled_top1"]["by_subject"]["1"], 0.005)
        self.assertEqual(result["verdict"], "bounded_null")

        specific = report([with_shuffles(i, 0.26) for i in range(10)], derived=derived)
        result = classify(specific, "edges", "80", sesoi=0.02, probe_chance=CHANCE)
        self.assertEqual(result["verdict"], "specific_effect")

    def test_size_matched_shuffles_can_replace_raw_shuffles_as_the_baseline(self):
        from eegdistill.verdict import classify

        def with_shuffles(i):
            v = view(target_top1=0.10, control_top1=0.28)
            for k in range(2):
                # Raw shuffled edits are small (top-1 0.26); scaled to the real
                # edit's size they do almost as much damage (0.105).
                s = view(target_top1=0.26, control_top1=0.28)["interventions"]["edges"]
                s["size_matched"] = {"top1": 0.105 + 0.001 * (i % 3), "margin": 0.0}
                v["interventions"][f"edges_shuf{k}"] = s
            return v

        derived = {f"edges_shuf{k}": {"kind": "shuffled", "from": "edges"} for k in range(2)}
        rep = report([with_shuffles(i) for i in range(10)], derived=derived)
        raw = classify(rep, "edges", "80", sesoi=0.02, probe_chance=CHANCE)
        self.assertEqual(raw["verdict"], "specific_effect")
        matched = classify(rep, "edges", "80", sesoi=0.02, probe_chance=CHANCE, size_matched=True)
        self.assertAlmostEqual(matched["excess_vs_size_matched_shuffled_top1"]["by_subject"]["1"], 0.005)
        self.assertEqual(matched["verdict"], "bounded_null")
        self.assertIsNotNone(raw["excess_vs_size_matched_shuffled_top1"])  # reported either way

    def test_shuffled_copies_are_not_scored_as_collateral_probes(self):
        from eegdistill.verdict import classify

        def with_shuffled_probe():
            v = view(target_top1=0.10, control_top1=0.28)
            edit = v["interventions"]["edges"]
            for scores in [v["baseline"], edit["target"], *edit["controls"]]:
                scores["probes"]["category_shuf0"] = {"balanced_accuracy": 0.3}
            return v

        r = report([with_shuffled_probe() for _ in range(10)],
                   derived={"category_shuf0": {"kind": "shuffled", "from": "category"}})
        result = classify(r, "edges", "80", sesoi=0.02, probe_chance=CHANCE)
        self.assertEqual(result["verdict"], "specific_effect")
        self.assertEqual(set(result["collateral_extra_loss"]), {"category"})

    def test_joint_target_counts_its_members_as_related_not_collateral(self):
        from eegdistill.verdict import classify

        def joint_view():
            v = view(target_top1=0.10, control_top1=0.28)
            for scores in [v["baseline"], v["interventions"]["edges"]["target"],
                           *v["interventions"]["edges"]["controls"]]:
                scores["probes"]["lowlevel"] = {"r2": scores["probes"]["edges"]["r2"]}
            v["interventions"]["lowlevel"] = v["interventions"].pop("edges")
            return v

        r = report([joint_view() for _ in range(10)],
                   derived={"lowlevel": {"kind": "joint", "of": ["edges"]}})
        result = classify(r, "lowlevel", "80", sesoi=0.02, probe_chance=CHANCE)
        self.assertEqual(result["verdict"], "specific_effect")
        self.assertEqual(set(result["collateral_extra_loss"]), {"category"})
        self.assertIn("edges", result["related_remaining"])

    def test_categorical_probe_without_chance_level_is_rejected(self):
        from eegdistill.verdict import classify

        with self.assertRaises(ValueError):
            classify(report([view(), view()]), "edges", "80", sesoi=0.02)


class PairedContrastTest(unittest.TestCase):
    def test_contrast_differences_two_properties_within_each_subject(self):
        from eegdistill.verdict import paired_contrast

        def two_edits(i):
            v = view(target_top1=0.10, control_top1=0.28)  # edges: D = 0.18
            cat = view(target_top1=0.20 + 0.01 * i, control_top1=0.28)  # D = 0.08 - 0.01 i
            v["interventions"]["cat_rank1"] = cat["interventions"]["edges"]
            return v

        r = report([two_edits(i) for i in range(10)])
        result = paired_contrast(r, "edges", "cat_rank1", "80")
        self.assertEqual(result["excess_top1_a_minus_b"]["n_subjects"], 10)
        self.assertAlmostEqual(result["excess_top1_a_minus_b"]["by_subject"]["1"], 0.10)
        self.assertAlmostEqual(result["excess_top1_a_minus_b"]["by_subject"]["10"], 0.19)
        self.assertGreater(result["excess_top1_a_minus_b"]["ci95"][0], 0)


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

    def test_size_matched_contrast_uses_shuffled_baseline_and_scales_by_baseline(self):
        from eegdistill.verdict import montage_contrast

        def montage(base, target, matched, n=4):
            views = []
            for _ in range(n):
                v = view(base_top1=base, target_top1=target, control_top1=base)
                s = view(target_top1=base, control_top1=base)["interventions"]["edges"]
                s["size_matched"] = {"top1": matched, "margin": 0.0}
                v["interventions"]["edges_shuf0"] = s
                views.append(v)
            return report(views, derived={"edges_shuf0": {"kind": "shuffled", "from": "edges"}})

        # Same 20% relative loss beyond controls, very different absolute sizes.
        occ = montage(base=0.50, target=0.30, matched=0.40)
        crown = montage(base=0.20, target=0.12, matched=0.16)
        result = montage_contrast(occ, crown, "edges", "80", size_matched=True)
        self.assertAlmostEqual(result["excess_top1_a"]["mean"], 0.10)
        self.assertAlmostEqual(result["excess_top1_b"]["mean"], 0.04)
        self.assertAlmostEqual(result["excess_top1_a_minus_b"]["mean"], 0.06)
        self.assertAlmostEqual(result["relative_excess_a"]["mean"], 0.20)
        self.assertAlmostEqual(result["relative_excess_b"]["mean"], 0.20)
        self.assertAlmostEqual(result["relative_excess_a_minus_b"]["mean"], 0.0)
        self.assertEqual(result["baseline"], "size_matched_shuffled")

    def test_contrast_rejects_reports_with_different_analysis_settings(self):
        from eegdistill.verdict import montage_contrast

        a = report([view(), view()])
        b = report([view(), view()], strength=0.5)
        with self.assertRaisesRegex(ValueError, "strength"):
            montage_contrast(a, b, "edges", "80")


class MergeReportsTest(unittest.TestCase):
    def test_per_subject_reports_merge_into_one_report_for_verdicts(self):
        from eegdistill.verdict import classify, merge_reports

        parts = [report([view(target_top1=0.10 + 0.01 * i, control_top1=0.28)]) for i in range(10)]
        for i, part in enumerate(parts, 1):  # one runner call per subject: --run i
            part["subjects"] = {str(i): part["subjects"]["1"]}
            part["summary"] = {"80": {"stale": True}}
        merged = merge_reports(parts)
        self.assertEqual(sorted(merged["subjects"], key=int), [str(i) for i in range(1, 11)])
        self.assertNotIn("summary", merged)  # per-subject summaries do not pool
        result = classify(merged, "edges", "80", sesoi=0.02, probe_chance=CHANCE)
        self.assertEqual(result["excess_top1"]["n_subjects"], 10)

    def test_merge_rejects_mismatched_settings_and_duplicate_subjects(self):
        from eegdistill.verdict import merge_reports

        with self.assertRaisesRegex(ValueError, "controls"):
            merge_reports([report([view()]), report([view()], controls=5)])
        with self.assertRaisesRegex(ValueError, "derived"):
            merge_reports([report([view()]), report([view()], derived={"x": {}})])
        with self.assertRaisesRegex(ValueError, "subject 1"):
            merge_reports([report([view()]), report([view()])])


if __name__ == "__main__":
    unittest.main()
