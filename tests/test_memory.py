import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import memory


def test_memory_save_and_search(tmp_path):
    test_db = tmp_path / "test_memory.db"

    original_database_path = memory.DATABASE_PATH
    memory.DATABASE_PATH = str(test_db)

    try:
        memory.init_db()

        test_text = "pytest test memory entry"

        memory.save_memory(test_text)

        results = memory.search_memory("pytest test memory entry")

        assert len(results) > 0
        assert any(test_text in row[1] for row in results)

    finally:
        memory.DATABASE_PATH = original_database_path