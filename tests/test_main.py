from voiceagent.__main__ import load_env


def test_load_env_reads_keys_without_overriding_shell(tmp_path):
    path = tmp_path / ".env"
    path.write_text('# comment\nGEMINI_API_KEY="abc"\n\nWHISPER_MODEL = medium\nSHELL_SET=file', encoding="utf-8")
    env = {"SHELL_SET": "shell"}
    load_env(path, env)
    assert env == {"GEMINI_API_KEY": "abc", "WHISPER_MODEL": "medium", "SHELL_SET": "shell"}


def test_load_env_missing_file_is_fine(tmp_path):
    env = {}
    load_env(tmp_path / "nope", env)
    assert env == {}
