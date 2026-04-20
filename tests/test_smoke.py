import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ingest.chunker import chunk
from ingest.loader import load_all, load_structured_records
from rag.memory import rewrite_query
from rag import pipeline
from rag.prompt import build_prompt
from rag import reranker
from rag.retriever import hybrid_retrieve


class SmokeTests(unittest.TestCase):
    def test_structured_loader_reads_crawler_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = {
                "type": "html",
                "url": "https://cukashmir.ac.in/#/publiczone",
                "title": "CUK Public Zone",
                "category": "general",
                "text": "Admissions 2026 are open now.",
                "outlinks": ["https://cukashmir.ac.in/#/admissions"],
                "notices": [{"text": "Admission notice", "link": "https://example.com/notice.pdf"}],
            }
            (root / "record.json").write_text(json.dumps(payload), encoding="utf-8")

            docs = load_structured_records(root)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0]["title"], "CUK Public Zone")
            self.assertTrue(any(link["url"] == "https://example.com/notice.pdf" for link in docs[0]["links"]))

    def test_structured_loader_includes_table_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = {
                "type": "html",
                "url": "https://cukashmir.ac.in/#/faculty",
                "title": "Faculty Directory",
                "category": "faculty",
                "text": "School staff directory.",
                "tables": [
                    [
                        ["Name", "Designation", "Email"],
                        ["Prof. Shahid Rasool", "Dean", "shahid@cukashmir.ac.in"],
                    ]
                ],
                "outlinks": [],
                "notices": [],
            }
            (root / "record.json").write_text(json.dumps(payload), encoding="utf-8")

            docs = load_structured_records(root)

            self.assertEqual(len(docs), 1)
            self.assertIn("Table 1:", docs[0]["text"])
            self.assertIn("Name: Prof. Shahid Rasool", docs[0]["text"])
            self.assertTrue(docs[0]["has_table"])

    def test_load_all_can_include_root_files_alongside_structured_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "structured").mkdir()
            (root / "manual").mkdir()

            structured_payload = {
                "type": "html",
                "url": "https://cukashmir.ac.in/#/admissions",
                "title": "Admissions",
                "category": "admissions",
                "text": "Structured crawler record.",
                "outlinks": [],
                "notices": [],
            }
            (root / "structured" / "record.json").write_text(json.dumps(structured_payload), encoding="utf-8")
            (root / "legacy-note.txt").write_text("Root level notice for students.", encoding="utf-8")

            fake_settings = SimpleNamespace(
                structured_dir=root / "structured",
                manual_dir=root / "manual",
                data_dir=root,
                include_manual_raw=False,
                include_legacy_root_raw=True,
            )

            with patch("ingest.loader.get_settings", return_value=fake_settings):
                docs = load_all(root)

            self.assertEqual(len(docs), 2)
            self.assertEqual({doc["source_kind"] for doc in docs}, {"crawler", "legacy_raw"})

    def test_chunker_preserves_titles_and_urls(self):
        docs = [
            {
                "text": "Admission notice for undergraduate programmes.",
                "links": [],
                "source": "https://cukashmir.ac.in/#/admissions",
                "source_path": "data/structured/sample.json",
                "source_url": "https://cukashmir.ac.in/#/admissions",
                "title": "Admissions",
                "category": "admissions",
                "file_type": "html",
                "doc_id": "doc-1",
                "source_kind": "crawler",
                "scraped_at": None,
                "ocr": False,
            }
        ]
        chunks = chunk(docs)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Title: Admissions", chunks[0]["text"])
        self.assertIn("URL: https://cukashmir.ac.in/#/admissions", chunks[0]["text"])

    def test_prompt_includes_source_numbers(self):
        prompt = build_prompt(
            "What is the admission process?",
            [
                {
                    "title": "Admissions",
                    "source_url": "https://cukashmir.ac.in/#/admissions",
                    "category": "admissions",
                    "text": "Admissions are open.",
                }
            ],
            [],
        )
        self.assertIn("[1]", prompt)
        self.assertIn("https://cukashmir.ac.in/#/admissions", prompt)
        self.assertIn("table rows", prompt.lower())

    def test_generation_models_include_distinct_gemini_fallbacks(self):
        settings = SimpleNamespace(
            generator_model="gemini-2.5-flash",
            fallback_generator_model="gemini-2.5-flash",
        )
        with patch("rag.pipeline.get_settings", return_value=settings):
            models = pipeline._generation_models()

        self.assertEqual(models[0], "gemini-2.5-flash")
        self.assertIn("gemini-2.5-flash-lite", models)
        self.assertIn("gemini-flash-latest", models)
        self.assertEqual(len(models), len(set(models)))

    def test_generation_models_preserve_configured_fallback_order(self):
        settings = SimpleNamespace(
            generator_model="gemini-2.5-flash",
            fallback_generator_model="gemini-3-flash-preview",
        )
        with patch("rag.pipeline.get_settings", return_value=settings):
            models = pipeline._generation_models()

        self.assertEqual(models[:2], ["gemini-2.5-flash", "gemini-3-flash-preview"])

    def test_memory_rewrite_resolves_recent_person_without_llm(self):
        history = [
            {
                "user": "who is the coordinator of media studies",
                "bot": "Prof. Shahid Rasool is the Dean, School of Media Studies.",
            }
        ]
        with patch("rag.memory.get_genai_client", return_value=None):
            rewritten = rewrite_query("what's his contact info", history)

        self.assertIn("Shahid Rasool", rewritten)
        self.assertNotIn(" his ", f" {rewritten.lower()} ")

    def test_memory_skips_llm_rewrite_for_standalone_question_with_history(self):
        history = [
            {
                "user": "who is the coordinator of media studies",
                "bot": "Prof. Shahid Rasool is the Dean, School of Media Studies.",
            }
        ]
        with patch("rag.memory.get_genai_client") as client_mock:
            rewritten = rewrite_query("who is shabir ahmad ahanger", history)

        self.assertEqual(rewritten, "who is shabir ahmad ahanger")
        client_mock.assert_not_called()

    def test_memory_local_rewrite_avoids_role_phrase_as_person(self):
        history = [
            {
                "user": "who is mr shabir ahmad ahanger",
                "bot": "Mr. (Dr.) Shabir Ahmad Ahanger is Associate Professor in the Department of Mathematics.",
            }
        ]
        with patch("rag.memory.get_genai_client") as client_mock:
            rewritten = rewrite_query("what is his contact info", history)

        self.assertIn("Shabir Ahmad Ahanger", rewritten)
        self.assertNotIn("Associate Professor's contact info", rewritten)
        client_mock.assert_not_called()

    def test_pipeline_uses_rewritten_query_for_generation_prompt(self):
        history = [{"user": "who is the coordinator of media studies", "bot": "Prof. Shahid Rasool is listed there."}]
        candidates = [
            {
                "title": "Administration",
                "source_url": "https://cukashmir.ac.in/#/administration",
                "source_path": "data/structured/administration.json",
                "category": "general",
                "text": "Prof. Shahid Rasool contact details.",
            }
        ]

        with patch("rag.pipeline.rewrite_query", return_value="What is Prof. Shahid Rasool's contact info?"), \
             patch("rag.pipeline.hybrid_retrieve", return_value=candidates), \
             patch("rag.pipeline.rerank", return_value=candidates), \
             patch("rag.pipeline.get_settings", return_value=SimpleNamespace(rerank_top_k=5, rerank_candidate_k=12)), \
             patch("rag.pipeline._generation_stream", return_value=iter([pipeline.TextChunk("ok")])) as generation_mock:
            stream, sources = pipeline.run("what's his contact info", history)

        self.assertEqual(list(stream)[0].text, "ok")
        self.assertEqual(len(sources), 1)
        prompt = generation_mock.call_args.args[0]
        self.assertIn("Question: What is Prof. Shahid Rasool's contact info?", prompt)

    def test_pipeline_uses_smaller_rerank_pool_for_exact_lookup(self):
        history = []
        candidates = [
            {
                "title": f"Result {index}",
                "source_url": f"https://cukashmir.ac.in/#/item-{index}",
                "source_path": f"data/structured/item-{index}.json",
                "category": "general",
                "text": f"Shabir Ahmad Ahanger record {index}",
            }
            for index in range(12)
        ]

        with patch("rag.pipeline.rewrite_query", return_value="who is mr shabir ahmad ahanger"), \
             patch("rag.pipeline.hybrid_retrieve", return_value=candidates), \
             patch("rag.pipeline.rerank", return_value=candidates[:5]) as rerank_mock, \
             patch("rag.pipeline.get_settings", return_value=SimpleNamespace(rerank_top_k=5, rerank_candidate_k=12)), \
             patch("rag.pipeline._generation_stream", return_value=iter([pipeline.TextChunk("ok")])):
            list(pipeline.run("who is mr shabir ahmad ahanger", history)[0])

        self.assertEqual(len(rerank_mock.call_args.args[1]), 8)

    def test_prompt_accepts_detailed_answer_style(self):
        prompt = build_prompt(
            "How do I apply?",
            [
                {
                    "title": "Admissions",
                    "source_url": "https://cukashmir.ac.in/#/admissions",
                    "category": "admissions",
                    "text": "Admissions are open with listed steps.",
                }
            ],
            [],
            answer_style="detailed",
        )

        self.assertIn("Give a short summary first", prompt)
        self.assertIn("Answer with citations:", prompt)

    def test_run_with_metadata_returns_source_preview_and_counts(self):
        history = []
        candidates = [
            {
                "title": "Admissions Notice",
                "source_url": "https://cukashmir.ac.in/#/admissions",
                "source_path": "data/structured/admissions.json",
                "category": "admissions",
                "text": "Admissions are open for 2026. Students must complete the online application form.",
                "matched_by": ["dense", "bm25"],
                "final_score": 0.42,
                "rerank_score": 2.1,
            }
        ]

        with patch("rag.pipeline.rewrite_query", return_value="admission process for 2026"), \
             patch("rag.pipeline.hybrid_retrieve", return_value=candidates), \
             patch("rag.pipeline.rerank", return_value=candidates), \
             patch("rag.pipeline.get_settings", return_value=SimpleNamespace(rerank_top_k=5, rerank_candidate_k=12)), \
             patch("rag.pipeline._generation_stream", return_value=iter([pipeline.TextChunk("ok")])):
            stream, sources, details = pipeline.run_with_metadata(
                "What is the admission process?",
                history,
                answer_style="balanced",
            )

        self.assertEqual(list(stream)[0].text, "ok")
        self.assertEqual(details["candidate_count"], 1)
        self.assertEqual(details["rerank_input_count"], 1)
        self.assertEqual(details["selected_chunk_count"], 1)
        self.assertEqual(details["rewritten_query"], "admission process for 2026")
        self.assertEqual(sources[0]["citation"], 1)
        self.assertIn("Admissions are open for 2026", sources[0]["preview"])
        self.assertEqual(sources[0]["matched_by"], ["dense", "bm25"])

    def test_staff_contact_queries_prefer_contact_records(self):
        settings = SimpleNamespace(retrieval_k=5)
        contact_item = {
            "chunk_id": "contact-1",
            "title": "Administration Contact",
            "source_url": "https://cukashmir.ac.in/#/administration/contact",
            "source_path": "data/structured/admin.json",
            "category": "contact",
            "text": "Prof. Shahid Rasool Dean School of Media Studies Email: shahid@cukashmir.ac.in Phone: 1234567890",
            "has_links": False,
            "links": [],
            "has_table": False,
            "table_row_count": 0,
            "contact_field_count": 2,
            "chunk_index": 0,
            "dense_score": 0.7,
        }
        policy_item = {
            "chunk_id": "policy-1",
            "title": "University Ordinance",
            "source_url": "https://cukashmir.ac.in/ordinance-19",
            "source_path": "data/structured/ordinance.json",
            "category": "general",
            "text": "The professor shall exercise competence and diligence in office.",
            "has_links": False,
            "links": [],
            "has_table": False,
            "table_row_count": 0,
            "contact_field_count": 0,
            "chunk_index": 0,
            "dense_score": 0.75,
        }

        with patch("rag.retriever.get_settings", return_value=settings), \
             patch("rag.retriever._dense_search", return_value=[contact_item, policy_item]), \
             patch("rag.retriever._bm25_search", return_value=[policy_item, contact_item]):
            results = hybrid_retrieve("what is Prof Shahid Rasool contact email", k=2)

        self.assertEqual(results[0]["chunk_id"], "contact-1")

    def test_rerank_keeps_relative_order_for_all_low_scores_without_warning(self):
        chunks = [
            {"title": "Alpha", "text": "alpha text"},
            {"title": "Beta", "text": "beta text"},
            {"title": "Gamma", "text": "gamma text"},
        ]
        mock_model = SimpleNamespace(predict=lambda pairs: [-3.8, -3.1, -4.2])

        with patch("rag.reranker.get_settings", return_value=SimpleNamespace(rerank_top_k=2)), \
             patch("rag.reranker._reranker", return_value=mock_model), \
             patch("rag.reranker.log.warning") as warning_mock, \
             patch("rag.reranker.log.info") as info_mock:
            results = reranker.rerank("latest btech result", chunks, top_k=2)

        self.assertEqual([item["title"] for item in results], ["Beta", "Alpha"])
        warning_mock.assert_not_called()
        info_mock.assert_called_once()

    def test_rerank_filters_low_scores_when_relevant_results_exist(self):
        chunks = [
            {"title": "Top", "text": "top text"},
            {"title": "Drop", "text": "drop text"},
            {"title": "Keep", "text": "keep text"},
        ]
        mock_model = SimpleNamespace(predict=lambda pairs: [0.9, -3.5, -1.0])

        with patch("rag.reranker.get_settings", return_value=SimpleNamespace(rerank_top_k=3)), \
             patch("rag.reranker._reranker", return_value=mock_model):
            results = reranker.rerank("admissions", chunks, top_k=3)

        self.assertEqual([item["title"] for item in results], ["Top", "Keep"])


if __name__ == "__main__":
    unittest.main()
