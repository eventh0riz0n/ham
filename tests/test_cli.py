import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def run_ham(db_path, *args):
    env = os.environ.copy()
    env["HAM_DB_PATH"] = str(db_path)
    env.pop("GEMINI_API_KEY", None)
    env.pop("GOOGLE_API_KEY", None)
    env.pop("OPENROUTER_API_KEY", None)
    return subprocess.run(
        [sys.executable, "-m", "ham.cli", *args],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_cli_list_show_delete_doctor_backup(tmp_path):
    db = tmp_path / "ham.db"
    result = run_ham(db, "remember", "cli managed memory", "--store", "semantic")
    assert result.returncode == 0, result.stderr
    chunk_id = [line.strip() for line in result.stdout.splitlines() if line.strip().startswith("semantic_")][0]

    result = run_ham(db, "list", "--store", "semantic")
    assert result.returncode == 0, result.stderr
    assert chunk_id in result.stdout

    result = run_ham(db, "show", chunk_id)
    assert result.returncode == 0, result.stderr
    assert "cli managed memory" in result.stdout

    backup = tmp_path / "backup.db"
    result = run_ham(db, "backup", "--out", str(backup))
    assert result.returncode == 0, result.stderr
    assert backup.exists()

    result = run_ham(db, "doctor")
    assert result.returncode == 0, result.stderr
    assert "schema_version" in result.stdout

    result = run_ham(db, "delete", chunk_id)
    assert result.returncode == 0, result.stderr
    assert "Deleted" in result.stdout


def test_cli_invalid_since_is_user_friendly(tmp_path):
    result = run_ham(tmp_path / "ham.db", "index", "--since", "not-a-date")
    assert result.returncode != 0
    assert "Invalid --since" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_migrate_dry_run_does_not_modify_v1_db(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL)")
    conn.execute("INSERT INTO schema_version(version, applied_at) VALUES (1, 1)")
    conn.execute(
        """CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            store TEXT NOT NULL,
            text TEXT NOT NULL,
            source TEXT NOT NULL,
            source_type TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            access_count INTEGER DEFAULT 0,
            last_accessed INTEGER,
            importance REAL DEFAULT 0.5,
            metadata TEXT DEFAULT '{}',
            UNIQUE(id)
        )"""
    )
    conn.commit()
    conn.close()

    result = run_ham(db, "migrate", "--dry-run")
    assert result.returncode == 0, result.stderr

    conn = sqlite3.connect(db)
    version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    columns = [row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()]
    conn.close()
    assert version == 1
    assert "embedding_provider" not in columns


def test_cli_backup_refuses_to_overwrite_without_force(tmp_path):
    db = tmp_path / "ham.db"
    run_ham(db, "remember", "backup safety", "--store", "semantic")
    backup = tmp_path / "backup.db"
    backup.write_text("existing")

    result = run_ham(db, "backup", "--out", str(backup))
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert backup.read_text() == "existing"

    result = run_ham(db, "backup", "--out", str(backup), "--force")
    assert result.returncode == 0, result.stderr
    assert backup.read_bytes() != b"existing"


def test_cli_consolidate_does_not_crash(tmp_path):
    db = tmp_path / "ham.db"
    result = run_ham(db, "consolidate")
    assert result.returncode == 0, result.stderr
    assert "Done." in result.stdout
