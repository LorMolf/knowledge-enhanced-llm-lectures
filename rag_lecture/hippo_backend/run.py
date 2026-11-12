"""Run the lecture's native HippoRAG retrieval in its isolated environment."""

import argparse
import hashlib
import ipaddress
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import Chunk, text_processing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw_input = args.input.read_bytes()
    task = json.loads(raw_input)
    if not isinstance(task, dict) or set(task) != {
        "corpus",
        "queries",
        "k",
        "llm_model",
        "llm_base_url",
        "llm_instance",
        "embedding_model",
    }:
        raise ValueError("Input must contain only the declared retrieval fields, not gold labels")
    docs, questions = task["corpus"], task["queries"]
    if not isinstance(docs, list) or not isinstance(questions, list) or not docs or not questions:
        raise ValueError("corpus and queries must be nonempty lists")
    for records, field in ((docs, "text"), (questions, "text")):
        if any(
            not isinstance(row, dict)
            or set(row) != {"id", field}
            or not isinstance(row["id"], str)
            or not row["id"].strip()
            or not isinstance(row[field], str)
            or not row[field].strip()
            for row in records
        ):
            raise ValueError("Every record must have a nonempty string id and text")
        if len({row["id"] for row in records}) != len(records):
            raise ValueError("Duplicate IDs are not allowed")
    if len({row["text"] for row in docs}) != len(docs):
        raise ValueError("Duplicate document text would collapse distinct source IDs")
    k = task["k"]
    if type(k) is not int or not 1 <= k <= len(docs):
        raise ValueError("k must be a positive integer no larger than the corpus")
    model, endpoint, instance = task["llm_model"], task["llm_base_url"], task["llm_instance"]
    if any(not isinstance(x, str) or not x.strip() for x in (model, endpoint, instance)):
        raise ValueError("LLM model, endpoint, and instance identity must be nonempty strings")
    url = urlsplit(endpoint)
    if (
        url.scheme != "http"
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Use a local, unauthenticated HTTP endpoint")
    if url.hostname != "localhost" and not ipaddress.ip_address(url.hostname).is_loopback:
        raise ValueError("LLM endpoint must be loopback")
    embedding = task["embedding_model"]
    if embedding != "BAAI/bge-base-en-v1.5":
        raise ValueError("This experiment uses the lecture's BGE baseline checkpoint")

    # Cache identity covers corpus IDs/text and the model/endpoint/instance, not query labels.
    identity = {
        "corpus": docs,
        "llm_model": model,
        "llm_base_url": endpoint,
        "llm_instance": instance,
        "embedding_model": embedding,
        "backend_revision": "d5c8329422e0a0b834a15874545cb6a74b4f9b26",
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    save_dir = Path("outputs") / f"hipporag-isolated-{fingerprint}"
    config = BaseConfig(openie_max_workers=1, retrieval_top_k=k)
    with HippoRAG(
        global_config=config,
        save_dir=str(save_dir),
        llm_model_name=model,
        llm_base_url=endpoint,
        embedding_model_name=embedding,
        embedding_provider="transformers",
        index_identity=fingerprint,
    ) as hippo:
        hippo.index([Chunk(content=row["text"], source_id=row["id"]) for row in docs])
        indexed = hippo.chunk_embedding_store.get_all_id_to_rows()
        expected = {row["id"]: row["text"] for row in docs}
        actual = {
            hippo.chunk_metadata[key]["source_id"]: row["content"] for key, row in indexed.items()
        }
        if len(indexed) != len(expected) or actual != expected:
            raise ValueError("Stored index does not match the requested corpus and IDs")
        texts = [row["text"] for row in questions]
        # Native scoring ranks the whole index before this return-depth slice;
        # requesting all documents leaves the scores unchanged and exposes cutoff ties.
        solutions = hippo.retrieve(texts, num_to_retrieve=len(docs))
        dense_solutions = hippo.retrieve_dpr(texts, num_to_retrieve=len(docs))
        if len(solutions) != len(questions) or len(dense_solutions) != len(questions):
            raise ValueError("Retrieval did not cover every query")
        results = []
        for query, graph, dense in zip(questions, solutions, dense_solutions):
            record = {"id": query["id"]}
            for name, solution in (("hipporag", graph), ("dense", dense)):
                if (
                    solution.question != query["text"]
                    or solution.doc_metadata is None
                    or len(solution.docs) != len(docs)
                    or len(solution.doc_scores) != len(docs)
                    or len(solution.doc_metadata) != len(docs)
                ):
                    raise ValueError(f"Incomplete or misaligned {name} result for {query['id']}")
                hits = []
                for text, score, metadata in zip(
                    solution.docs, solution.doc_scores, solution.doc_metadata
                ):
                    doc_id = metadata.get("source_id")
                    if (
                        doc_id not in expected
                        or expected[doc_id] != text
                        or not math.isfinite(float(score))
                    ):
                        raise ValueError(f"Invalid {name} result document or score: {doc_id}")
                    hits.append({"corpus_id": doc_id, "score": float(score)})
                if len({hit["corpus_id"] for hit in hits}) != len(docs):
                    raise ValueError(f"Duplicate result IDs for {query['id']}")
                hits.sort(key=lambda hit: (-hit["score"], hit["corpus_id"]))
                record[name] = hits[:k]
            if graph.graph_seeds is None:
                raise ValueError("HippoRAG did not report graph seeds; retrieval mode is unknown")
            record["mode"] = "graph_seeded" if len(graph.graph_seeds) else "dense_fallback"
            record["graph_seeds"] = [list(seed) for seed in graph.graph_seeds]
            if query is questions[0]:
                record["seed_evidence"] = []
                for seed in graph.graph_seeds:
                    key = str(tuple(text_processing(list(seed))))
                    chunk_ids = hippo.proc_triples_to_docs.get(key, set())
                    if not chunk_ids:
                        raise ValueError(f"No source documents for graph seed {seed}")
                    source_ids = sorted(
                        hippo.chunk_metadata[chunk_id]["source_id"] for chunk_id in chunk_ids
                    )
                    record["seed_evidence"].append(
                        {
                            "seed": list(seed),
                            "sources": [
                                {"id": doc_id, "text": expected[doc_id]} for doc_id in source_ids
                            ],
                        }
                    )
            results.append(record)
        output = {
            "input_sha256": hashlib.sha256(raw_input).hexdigest(),
            "index_fingerprint": fingerprint,
            "graph_info": hippo.get_graph_info(),
            "results": results,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
