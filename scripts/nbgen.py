"""Tiny notebook builder + executor for the mmpartnet result notebooks. Produces a notebook in the shared
dist style: (1) a top markdown cell stating the ONE question, (2) thin code cells (load JSON + viz), (3) a
results-analysis markdown cell that pulls numbers from the loaded JSON. Writes a SOURCE notebook (outputs
stripped) and an EXECUTED copy (run with nbclient, outputs embedded) -- "recovering the executed versions".

  from nbgen import md, code, build
  build(src_path, exec_path, [md("# title"), code("..."), md("## results")])
"""
from __future__ import annotations
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient


def md(text):
    return ("md", text)


def code(text):
    return ("code", text)


def build(src_path, exec_path, cells, timeout=600, kernel="python3"):
    nb = new_notebook()
    nb.cells = [new_markdown_cell(t) if k == "md" else new_code_cell(t) for k, t in cells]
    nb.metadata["kernelspec"] = {"name": kernel, "display_name": "Python 3", "language": "python"}
    Path(src_path).parent.mkdir(parents=True, exist_ok=True)
    Path(exec_path).parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(src_path))                          # source (outputs stripped)
    ok, err = True, None
    try:
        NotebookClient(nb, timeout=timeout, kernel_name=kernel,
                       resources={"metadata": {"path": str(Path(src_path).parent)}}).execute()
    except Exception as e:                                     # keep partial outputs; report failure
        ok, err = False, str(e)[:300]
    nbformat.write(nb, str(exec_path))                         # executed copy (with outputs)
    print(f"notebook: {src_path}\nexecuted: {exec_path}  ({'OK' if ok else 'EXEC FAILED: ' + err})")
    return ok
