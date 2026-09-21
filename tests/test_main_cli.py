"""main.py CLI 進入點的測試。

重點是「設定檔壞掉時要給清楚的訊息與結束碼」，而不是噴 traceback。
"""

from __future__ import annotations

import json

import pytest

import main


@pytest.mark.parametrize(
    "payload, reason",
    [
        ("[1, 2, 3]", "最外層不是物件"),
        ('{"rl": [1]}', "區塊不是物件"),
        ('{"rl": []}', "區塊是空陣列（falsey，以前會被安靜忽略）"),
        ("{ 這不是 JSON", "JSON 語法錯誤"),
        ('{"rl": {"learning_rate": -5}}', "數值不合法"),
        ('{"rl": {"dropout": 2}}', "dropout 超出範圍"),
    ],
)
def test_bad_config_exits_with_code_2(tmp_path, capsys, monkeypatch, payload, reason):
    """壞掉的設定檔要回傳 2 並印出一行說明。

    SystemConfig.load() 以前放在 try 之外，所以最外層不是物件時
    會直接把 TypeError 冒出來變成 traceback（結束碼 1）。
    """
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")

    exit_code = main.main(["--mode", "train", "--config", str(path)])

    assert exit_code == 2, f"{reason} 應該回傳 2"
    assert "設定檔有誤" in capsys.readouterr().err


def test_missing_config_file_uses_defaults(tmp_path, monkeypatch):
    """--config 指到不存在的檔案時退回預設值（load() 的既有行為）。"""
    monkeypatch.chdir(tmp_path)
    parser = main.build_parser()
    args = parser.parse_args(["--config", str(tmp_path / "nope.json")])
    assert args.config.endswith("nope.json")


def test_valid_config_is_accepted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "ok.json"
    path.write_text(json.dumps({"ui": {"chart_height": 420}}), encoding="utf-8")

    from config import SystemConfig

    config = SystemConfig.load(path)
    config.validate()
    assert config.ui.chart_height == 420


def test_help_does_not_crash(capsys):
    with pytest.raises(SystemExit) as exc:
        main.main(["--help"])
    assert exc.value.code == 0
    assert "--mock-env" in capsys.readouterr().out


def test_vercel_app_root_returns_html():
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = dict(headers)

    response = b"".join(main.app({"PATH_INFO": "/"}, start_response)).decode("utf-8")

    assert captured["status"] == "200 OK"
    assert captured["headers"]["Content-Type"].startswith("text/html")
    assert "SmartTraffic Multimodal AI" in response
    assert "python main.py --mode web" in response


def test_vercel_app_health_returns_json():
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = dict(headers)

    response = b"".join(main.app({"PATH_INFO": "/health"}, start_response)).decode("utf-8")

    assert captured["status"] == "200 OK"
    assert captured["headers"]["Content-Type"].startswith("application/json")
    assert response == '{"status":"ok","service":"smarttraffic-multimodal-ai"}'
