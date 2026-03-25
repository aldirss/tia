import os
from pathlib import Path
import yaml


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


class ConfigError(Exception):
    pass


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise ConfigError(f"Arquivo não encontrado: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Erro ao ler YAML: {e}") from e

    if not config:
        raise ConfigError("Arquivo de configuração vazio.")

    return config

def set_env_variables(config: dict) -> None:
    api_keys = config.get("api_keys", {})

    mapping = {
        "tavily": "TAVILY_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }

    for yaml_key, env_name in mapping.items():
        value = api_keys.get(yaml_key)
        if value and isinstance(value, str) and value.strip():
            os.environ[env_name] = value.strip()


config = load_config()
set_env_variables(config)

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

