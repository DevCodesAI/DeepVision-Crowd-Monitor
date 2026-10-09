import ast
from pathlib import Path

def test_python_files_parse():
    root = Path(__file__).resolve().parents[1]
    for path in [root/'app.py', root/'deepvision/model.py', root/'deepvision/inference.py']:
        ast.parse(path.read_text(encoding='utf-8'))
