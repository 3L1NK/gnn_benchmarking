from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_audit(thesis_dir: Path) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "scripts/audit_bibliography.py", "--thesis-dir", str(thesis_dir)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )


def test_bibliography_audit_fails_on_placeholder_metadata(tmp_path):
    thesis_dir = tmp_path / "thesis"
    (thesis_dir / "chapters").mkdir(parents=True, exist_ok=True)
    (thesis_dir / "chapters" / "01_intro.tex").write_text(
        "Intro cites \\citep{foo}.\n",
        encoding="utf-8",
    )
    (thesis_dir / "references.bib").write_text(
        """
@misc{foo,
  author={Unknown},
  title={Foo},
  note={Local source file in repository papers folder},
  year={2020}
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    proc = _run_audit(thesis_dir)
    assert proc.returncode != 0
    assert "FAILED" in proc.stdout


def test_bibliography_audit_passes_for_clean_refs(tmp_path):
    thesis_dir = tmp_path / "thesis"
    (thesis_dir / "chapters").mkdir(parents=True, exist_ok=True)
    (thesis_dir / "chapters" / "01_intro.tex").write_text(
        "Intro cites \\citep{foo}.\n",
        encoding="utf-8",
    )
    (thesis_dir / "references.bib").write_text(
        """
@article{foo,
  author={Doe, Jane},
  title={A Proper Reference},
  journal={Journal of Tests},
  year={2020}
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    proc = _run_audit(thesis_dir)
    assert proc.returncode == 0
    assert "OK" in proc.stdout
