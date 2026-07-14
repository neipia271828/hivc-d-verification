from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HTML_PATH = REPO_ROOT / "scripts" / "local_preview.html"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required to test browser JavaScript")
def test_csv_parser_preserves_newlines_inside_quoted_fields() -> None:
    source = HTML_PATH.read_text(encoding="utf-8")
    start = source.index("function parseCSV(text) {")
    end = source.index("\nfunction normalizeRecord", start)
    node_script = """
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf('function parseCSV(text) {');
const end = html.indexOf('\\nfunction normalizeRecord', start);
eval(html.slice(start, end));
process.stdout.write(JSON.stringify(parseCSV(fs.readFileSync(0, 'utf8'))));
"""
    csv_text = (
        "turn,alpha_vote_thinking,discussion_transcript\r\n"
        '0,"first line\nsecond line","[{""speaker"":""alpha""}]"\r\n'
        '1,"one line","[{""speaker"":""beta""}]"\r\n'
    )
    result = subprocess.run(
        ["node", "-e", node_script, str(HTML_PATH)],
        input=csv_text,
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(result.stdout) == [
        {
            "turn": "0",
            "alpha_vote_thinking": "first line\nsecond line",
            "discussion_transcript": '[{"speaker":"alpha"}]',
        },
        {
            "turn": "1",
            "alpha_vote_thinking": "one line",
            "discussion_transcript": '[{"speaker":"beta"}]',
        },
    ]
