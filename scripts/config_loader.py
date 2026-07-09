"""実験スクリプト用の YAML 設定ローダー。

configファイル（YAML）を既定値とし、CLI引数はconfigより優先して上書きする。
PyYAML が無い場合は JSON 互換のサブセットでフォールバックするが、基本は PyYAML を
前提とする（requirements に追加推奨）。

使い方:
    from config_loader import build_argparser_from_config, load_config

    parser = build_argparser_from_config("configs/experiment.yaml")
    args = parser.parse_args()
    # args は config の値で埋まり、CLIで指定した項目だけ上書きされる
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """PyYAMLなしでフラットな key: value YAMLをパースする。

    ネストした構造やアンカー等はサポートしない。gpu_server.yaml のような
    単純な設定ファイル用のフォールバック。
    """
    result: dict[str, Any] = {}
    current_list_key: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        # コメント行・空行をスキップ
        if not stripped or stripped.startswith("#"):
            continue
        # リスト要素（- value）
        if stripped.startswith("- ") and current_list_key is not None:
            val = stripped[2:].strip()
            if not isinstance(result.get(current_list_key), list):
                result[current_list_key] = []
            result[current_list_key].append(_parse_scalar(val))
            continue
        # key: value
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                # 次行以降がリストの可能性
                current_list_key = key
                result[key] = []
            else:
                current_list_key = None
                result[key] = _parse_scalar(val)
    return result


def _parse_scalar(val: str) -> Any:
    """YAMLスカラー値をPython型に変換。"""
    if val in ("null", "~", "None"):
        return None
    if val in ("true", "True", "yes"):
        return True
    if val in ("false", "False", "no"):
        return False
    # 整数
    try:
        return int(val)
    except ValueError:
        pass
    # 浮動小数
    try:
        return float(val)
    except ValueError:
        pass
    # 文字列（クォート除去）
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    return val


def load_yaml(path: Path) -> dict[str, Any]:
    """YAMLファイルを読み込む。PyYAMLが無い場合は簡易パーサーでフォールバック。"""
    text = path.read_text(encoding="utf-8")
    if _HAS_YAML:
        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    # フォールバック1: JSONとしてパスを試みる
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        pass
    # フォールバック2: 簡易YAMLパーサー
    return _parse_simple_yaml(text)


def resolve_path(value: str | None) -> str | None:
    """相対パスをリポジトリルート基準で解決した絶対パス文字列を返す。"""
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p)


def _coerce(value: Any, target_type: type) -> Any:
    """config値をCLI引数の型に合わせる。"""
    if value is None:
        return None
    if target_type is bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes", "on")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is list:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return [value]
    return str(value)


def build_argparser_from_config(
    config_path: str | Path | None,
    description: str = "",
    extra_args: list[dict] | None = None,
) -> tuple[argparse.ArgumentParser, dict[str, Any]]:
    """configファイルを読み込み、その値をdefaultとしたArgumentParserを返す。

    extra_args: configに含まれない追加引数を [{name, kwargs}, ...] 形式で渡せる。
    戻り値: (parser, config_dict)
    """
    config: dict[str, Any] = {}
    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists():
            config = load_yaml(path)

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None,
                        help="YAML設定ファイルパス（指定時はその値で上書きされる）")

    # configの全キーを引数として登録（kebab-caseに変換）
    for key, value in config.items():
        arg_name = "--" + key.replace("_", "-")
        # 型推定
        if isinstance(value, bool):
            arg_type = bool
            # boolは store_true/store_false ではなく直接値を受け取る形に
            parser.add_argument(arg_name, dest=key, default=value, type=str,
                                help=f"(config既定値: {value}) true/false")
        elif isinstance(value, int) and not isinstance(value, bool):
            parser.add_argument(arg_name, dest=key, default=value, type=int,
                                help=f"(config既定値: {value})")
        elif isinstance(value, float):
            parser.add_argument(arg_name, dest=key, default=value, type=float,
                                help=f"(config既定値: {value})")
        elif isinstance(value, list):
            parser.add_argument(arg_name, dest=key, default=value, nargs="*",
                                help=f"(config既定値: {value})")
        else:
            parser.add_argument(arg_name, dest=key, default=value, type=str,
                                help=f"(config既定値: {value})")

    # 追加引数（configに無いもの）
    if extra_args:
        for ea in extra_args:
            parser.add_argument(ea["name"], **ea.get("kwargs", {}))

    return parser, config


def apply_config_overrides(
    args: argparse.Namespace,
    config: dict[str, Any],
    arg_specs: dict[str, type],
) -> argparse.Namespace:
    """CLIで明示的に指定された引数のみを優先し、未指定はconfig値を使う。

    argparse は default を設定すると「未指定」と「指定」を区別できないため、
    この関数では「CLI値 == config値」の場合はconfig値を採用（実質同じ）とし、
    「CLI値 != config値」の場合はCLI値を採用する単純な上書き方式をとる。
    bool型の文字列 "true"/"false" も適切に変換する。
    """
    for key, spec_type in arg_specs.items():
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            setattr(args, key, _coerce(cli_value, spec_type))
        elif key in config:
            setattr(args, key, _coerce(config[key], spec_type))
    return args


def merge_config_and_cli(
    config_path: str | Path | None,
    cli_overrides: dict[str, Any],
    defaults: dict[str, Any],
    types: dict[str, type],
) -> dict[str, Any]:
    """config → CLI の順で優先される設定dictを返す（シンプル版）。

    config_path が与えられた場合はそのconfigで defaults を上書きし、
    さらに cli_overrides（None以外の値のみ）で上書きする。
    """
    result = dict(defaults)
    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists():
            config = load_yaml(path)
            for key, value in config.items():
                if key in types:
                    result[key] = _coerce(value, types[key])
                else:
                    result[key] = value
    for key, value in cli_overrides.items():
        if value is not None:
            result[key] = _coerce(value, types.get(key, str)) if key in types else value
    return result
