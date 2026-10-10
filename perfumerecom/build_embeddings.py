"""Precompute the perfume embeddings that Perfume_Recommender_Decision1.ipynb downloads.

Runs the notebook's own data and embedding cells, so the cached file and its fingerprint always match
what the notebook computes. Writes perfume_embeddings_<model>.npy and .npy.sha256 next to this script.
Used by .github/workflows/perfume-embeddings.yml; can also be run by hand:  python perfumerecom/build_embeddings.py
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "Perfume_Recommender_Decision1.ipynb"


def cell_containing(cells, marker):
    for cell in cells:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and marker in source:
            return source
    raise RuntimeError(f"No code cell in {NOTEBOOK.name} contains {marker!r}")


def main():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    data_cell = cell_containing(cells, "find_or_download_data()")
    embedding_cell = cell_containing(cells, "EMBEDDINGS_PATH")

    os.chdir(HERE)  # the notebook reads fragrantica.xlsx and writes the cache in its own folder
    namespace = {"os": os}
    exec(compile(data_cell, "data cell", "exec"), namespace)
    exec(compile(embedding_cell, "embedding cell", "exec"), namespace)
    print(f"Wrote {namespace['EMBEDDINGS_PATH']} for {len(namespace['doc_embeddings']):,} perfumes")


if __name__ == "__main__":
    main()
