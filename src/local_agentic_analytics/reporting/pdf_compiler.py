"""Compile LaTeX files into PDF when a local compiler is available."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def compile_pdf(tex_path: str | Path, output_dir: str | Path) -> Path:
    tex_file = Path(tex_path).resolve()
    if not tex_file.exists():
        raise FileNotFoundError(f"LaTeX file not found: {tex_file}")

    pdf_dir = Path(output_dir).resolve()
    pdf_dir.mkdir(parents=True, exist_ok=True)

    tectonic_path = shutil.which("tectonic")
    pdflatex_path = shutil.which("pdflatex")

    if tectonic_path:
        command = [
            tectonic_path,
            "--outdir",
            str(pdf_dir),
            tex_file.name,
        ]
        compiler_name = "tectonic"
    elif pdflatex_path:
        command = [
            pdflatex_path,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(pdf_dir),
            tex_file.name,
        ]
        compiler_name = "pdflatex"
    else:
        raise RuntimeError(
            "No LaTeX compiler found. Install tectonic or pdflatex to generate PDF."
        )

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        cwd=str(tex_file.parent),
    )
    if result.returncode != 0:
        log = _compact_compile_log(result.stdout, result.stderr)
        raise RuntimeError(f"{compiler_name} failed to compile PDF.\n{log}")

    pdf_path = pdf_dir / f"{tex_file.stem}.pdf"
    if not pdf_path.exists():
        raise RuntimeError(
            f"{compiler_name} finished without creating expected PDF: {pdf_path}"
        )

    return pdf_path


def _compact_compile_log(stdout: str, stderr: str, max_lines: int = 25) -> str:
    combined = "\n".join(part for part in [stdout, stderr] if part).strip()
    if not combined:
        return "No compiler output was captured."

    lines = combined.splitlines()
    return "\n".join(lines[-max_lines:])
