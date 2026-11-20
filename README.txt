Copy the files into your repo (same relative paths), then add this line to .env:

    POLICY_CHECK=1

New: src/discovery_agent/policy_check.py, tests/test_policy_check.py
Changed: constraints.py, config.py, graph.py, ui/streamlit_app.py
(changes-to-existing-files.diff shows exactly what changed if you have local edits to merge.)
Your files use Windows (CRLF) line endings; the changed files keep them.
