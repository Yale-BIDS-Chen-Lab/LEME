import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.metrics import percentile_intervals, score_predictions
from evaluation.score import main


class MetricTests(unittest.TestCase):
    def test_empty_responses_are_counted_and_text_is_preserved(self):
        rows = [{"id": "one", "reference": " patient ", "prediction": " patients "},
                {"id": "two", "reference": "patient", "prediction": ""}]
        summary, scores = score_predictions(rows, metrics=["rougeL"])
        self.assertEqual(summary["examples"], 2)
        self.assertEqual(summary["empty_predictions"], 1)
        self.assertEqual(summary["rougeL_f1"], 0.5)
        self.assertEqual([row["rougeL_f1"] for row in scores], [1.0, 0.0])
        self.assertEqual(rows[0]["prediction"], " patients ")

    @patch("evaluation.metrics._bert_scores")
    def test_f1_is_the_average_of_case_f1_not_averaged_precision_recall(self, bert):
        bert.return_value = ({"bertscore_precision": [1.0, 0.2],
                              "bertscore_recall": [0.2, 1.0],
                              "bertscore_f1": [1 / 3, 1 / 3]}, {"hash": "test"})
        rows = [{"id": "z", "reference": " A ", "prediction": " B "},
                {"id": "a", "reference": "C", "prediction": "D"}]
        summary, scores = score_predictions(rows, metrics=["bertscore"], device="cpu")
        self.assertAlmostEqual(summary["bertscore_f1"], 1 / 3)
        self.assertEqual([row["id"] for row in scores], ["z", "a"])
        bert.assert_called_once_with([" B ", "D"], [" A ", "C"], 64, "cpu")

    @patch("evaluation.metrics._bert_scores")
    def test_invalid_reference_and_duplicate_ids_fail_before_model_loading(self, bert):
        cases = [[], [{"reference": "", "prediction": "A"}],
                 [{"reference": "A", "prediction": None}],
                 [{"id": "same", "reference": "A", "prediction": "A"}] * 2]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                score_predictions(rows)
        bert.assert_not_called()

    @patch("evaluation.metrics._bert_scores")
    def test_incomplete_or_nonfinite_metric_vectors_are_rejected(self, bert):
        rows = [{"reference": "A", "prediction": "A"}]
        for values in ([], [float("nan")]):
            bert.return_value = ({"bertscore_f1": values}, {})
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "finite"):
                score_predictions(rows, metrics=["bertscore"])

    def test_bootstrap_does_not_replace_the_original_mean(self):
        rows = [{"reference": "cat", "prediction": prediction}
                for prediction in ("cat", "cat", "")]
        first, _ = score_predictions(rows, metrics=["rougeL"], bootstrap_samples=37, seed=7)
        second, _ = score_predictions(rows, metrics=["rougeL"], bootstrap_samples=37, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["rougeL_f1"], 2 / 3)
        self.assertEqual(first["bootstrap"]["method"], "percentile")
        self.assertEqual(first["bootstrap"]["intervals"]["rougeL_f1"], {"low": 0.0, "high": 1.0})

    def test_constant_scores_have_a_degenerate_percentile_interval(self):
        self.assertEqual(percentile_intervals({"score": [0.75]}, 100, 0),
                         {"score": {"low": 0.75, "high": 0.75}})


class ScoringCommandTests(unittest.TestCase):
    def test_saved_jsonl_is_scored_without_overwriting_inputs_or_inference_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            path.write_text(json.dumps({"id": "one", "reference": "cat", "prediction": "cat"}) + "\n")
            original = path.read_bytes()
            inference_metrics = path.with_suffix(".metrics.json")
            inference_metrics.write_text('{"model": "example"}\n')
            with patch("sys.argv", ["score", "--predictions", str(path), "--metrics", "rougeL"]), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads(path.with_suffix(".scores.json").read_text())
            self.assertEqual(result["rougeL_f1"], 1.0)
            self.assertEqual(result["prediction_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(json.loads(inference_metrics.read_text()), {"model": "example"})
            self.assertEqual(len(path.with_suffix(".scores.jsonl").read_text().splitlines()), 1)

    def test_output_paths_cannot_overwrite_the_prediction_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            path.write_text("untouched\n")
            for output in (path, path.with_suffix(".json")):
                with self.subTest(output=output), patch("sys.argv", ["score", "--predictions", str(path), "--output", str(output)]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main()
                self.assertEqual(path.read_text(), "untouched\n")


if __name__ == "__main__":
    unittest.main()
