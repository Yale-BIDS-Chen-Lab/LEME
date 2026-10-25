import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.data import load_benchmark, normalize_record
from leme.data_generation.client import azure_client, json_response, score_value
from leme.data_generation.generate_instructions import generated_rows, generation_prompt
from leme.data_generation.pair_generation import make_pair, preference_strategy
from leme.data_generation.refine_instructions import refine_row
from leme.data_generation.score_responses import score_row
from leme.io import jsonl_writer, read_records
from leme.training.prepare_data import dataset_info


class BenchmarkCompatibilityTests(unittest.TestCase):
    def test_released_schema_preserves_text_and_pmid(self):
        row = {"id": "title-1", "input": "Abstract\ntext", "reference": "Title", "source": "PubMed", "pmid": "123"}
        self.assertEqual(normalize_record(row, "title_prediction", 0), row)

    def test_legacy_formats_map_to_same_record(self):
        variants = [
            ("patient_qa", {"query": "Question", "answer": "Answer"}),
            ("title_prediction", {"question": "Question", "answer": "Answer"}),
            ("title_prediction", {"predict_title": {"question": "Question", "answer": "Answer"}}),
            ("treatment_planning", {"input": "Question", "output": "Answer"}),
        ]
        for task, row in variants:
            with self.subTest(task=task, row=row):
                normalized = normalize_record(row, task, 0)
                self.assertEqual(normalized["input"], "Question")
                self.assertEqual(normalized["reference"], "Answer")

    def test_restricted_tasks_require_local_data(self):
        for task in ("patient_qa", "treatment_planning"):
            with self.assertRaisesRegex(ValueError, "--data-file"):
                load_benchmark(task)

    def test_missing_reference_fails_before_inference(self):
        with self.assertRaisesRegex(ValueError, "reference"):
            normalize_record({"input": "Question"}, "patient_consultation", 0)


class InstructionGenerationTests(unittest.TestCase):
    def test_clinical_examples_retain_context_without_duplicate_rows(self):
        questions = [{"question": "Summarize", "answer": "A"}, {"question": "Extract", "answer": "B"}]
        result = generated_rows("clinical", {"questions": questions}, "Case context", 3)
        self.assertEqual(len(result), 2)
        self.assertNotEqual(result[0]["id"], result[1]["id"])
        self.assertTrue(all("Case context" in row["question"] for row in result))

    def test_usmle_prompt_requires_and_inserts_examples(self):
        with self.assertRaisesRegex(ValueError, "examples"):
            generation_prompt("usmle", "Case", random.Random(0))
        examples = [{"question": f"Question {i}", "answer": "A", "options": dict.fromkeys("ABCD", "Option")}
                    for i in range(3)]
        prompt = generation_prompt("usmle", "Case", random.Random(0), examples)
        self.assertNotIn("{question_", prompt)
        self.assertNotIn("{op_", prompt)
        self.assertTrue(prompt.endswith("Case"))

    @patch("leme.data_generation.refine_instructions.complete")
    def test_multiple_choice_retains_only_matching_answers(self, complete):
        row = {"id": "mc-1", "question": "Which option?", "answer": "B"}
        complete.return_value = "## Explanation: Reasoning\n## Answer: A"
        self.assertIsNone(refine_row(row, "multiple_choice", None, "test"))
        complete.return_value = "## Explanation: Reasoning\n## Answer: B"
        result = refine_row(row, "multiple_choice", None, "test")
        self.assertEqual(result["conversations"][0]["value"], row["question"])
        self.assertIn("Reasoning", result["conversations"][1]["value"])
        self.assertTrue(result["conversations"][1]["value"].endswith("## Answer: B"))

    @patch("leme.data_generation.refine_instructions.complete", return_value="ERROR")
    def test_flawed_open_question_is_excluded(self, complete):
        self.assertIsNone(refine_row({"id": "q", "question": "Q", "answer": "A"}, "open_ended", None, "test"))


class PreferenceConstructionTests(unittest.TestCase):
    def setUp(self):
        self.row = {"input": "Question", "response": "Original", "truthfulness": "# Score: 5\n# Explanation: Correct",
                    "informativeness": "# Score: 5\n# Explanation: Complete"}

    def test_strategy_uses_truthfulness_before_informativeness(self):
        row = {**self.row, "truthfulness": "# Score: 3", "informativeness": None}
        self.assertEqual(preference_strategy(row, Mock()), ("restore", False, "# Score: 3"))
        rng = Mock()
        rng.random.return_value = 0.9
        self.assertEqual(preference_strategy(self.row, rng)[:2], ("hallucinate", True))
        rng.random.return_value = 0.1
        self.assertEqual(preference_strategy(self.row, rng)[:2], ("deinform", True))
        self.assertEqual(preference_strategy({**self.row, "informativeness": "# Score: 2"}, rng)[:2], ("enrich", False))

    @patch("leme.data_generation.pair_generation.complete", return_value="Improved")
    def test_restored_response_becomes_chosen(self, complete):
        pair = make_pair({**self.row, "truthfulness": "# Score: 2", "informativeness": None}, None, "test", random.Random(0))
        self.assertEqual(pair["chosen"]["value"], "Improved")
        self.assertEqual(pair["rejected"]["value"], "Original")
        self.assertEqual(pair["conversations"], [{"from": "human", "value": "Question"}])

    @patch("leme.data_generation.score_responses.complete", return_value="# Score: 3\n# Explanation: Incorrect")
    def test_untruthful_responses_are_not_scored_for_informativeness(self, complete):
        result = score_row({"input": "Question", "prediction": "Answer"}, None, "test")
        self.assertIsNone(result["informativeness"])
        complete.assert_called_once()

    def test_invalid_scores_are_rejected(self):
        for value in (None, "# Score: 15", "# Score: 0", "No score"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                score_value(value)


class LocalDataTests(unittest.TestCase):
    def test_completed_api_records_survive_interruption(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "records.jsonl"
            with self.assertRaises(RuntimeError):
                with jsonl_writer(path) as emit:
                    emit({"text": "Completed record"})
                    raise RuntimeError("Interrupted")
            self.assertEqual(read_records(path), [{"text": "Completed record"}])

    def test_code_fenced_json_is_supported(self):
        self.assertEqual(json_response('```json\n{"questions": []}\n```'), {"questions": []})

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_credentials_fail_without_sdk_or_network(self):
        with self.assertRaisesRegex(ValueError, "AZURE_OPENAI_API_KEY"):
            azure_client()

    def test_training_metadata_maps_the_released_message_format(self):
        info = dataset_info()["leme_dpo"]
        self.assertTrue(info["ranking"])
        self.assertEqual(info["columns"], {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"})
        self.assertEqual(info["tags"]["content_tag"], "value")


if __name__ == "__main__":
    unittest.main()
