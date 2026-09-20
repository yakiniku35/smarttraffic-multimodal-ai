"""config.py 的單元測試。"""

from __future__ import annotations

import json

import pytest

from config import (
    MultimodalConfig,
    RLConfig,
    SystemConfig,
    TrafficConfig,
    UIConfig,
    load_env_file,
)


def test_default_config_is_valid():
    """預設設定本身就應該是合法的。"""
    SystemConfig().validate()


def test_sub_configs_are_independent():
    """每個 SystemConfig 實例都要有自己的子設定物件。

    這是 dataclass 用 field(default_factory=...) 要防的經典陷阱：
    直接寫 `multimodal: MultimodalConfig = MultimodalConfig()` 的話，
    所有實例會共用同一個物件，改了一個就會全部一起變。
    """
    a, b = SystemConfig(), SystemConfig()
    a.rl.learning_rate = 0.5

    assert b.rl.learning_rate != 0.5
    assert a.multimodal is not b.multimodal


def test_traffic_config_has_use_gui():
    """sumo_interface.py 會讀 config.use_gui，之前這個欄位不存在。"""
    assert hasattr(TrafficConfig(), "use_gui")
    assert TrafficConfig().use_gui is False


def test_rl_config_has_state_and_action_dim():
    """state_dim / action_dim 以前是在 main.py 動態塞進去的。"""
    rl = RLConfig()
    assert rl.state_dim > 0
    assert rl.action_dim > 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"embedding_dim": 0}, "embedding_dim"),
        ({"embedding_dim": 100, "num_attention_heads": 8}, "整除"),
        ({"dropout": 1.5}, "dropout"),
    ],
)
def test_multimodal_validation_rejects_bad_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MultimodalConfig(**kwargs).validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": -1},
        {"batch_size": 0},
        {"clip_range": 1.5},
        {"gamma": 0},
        {"state_dim": 0},
    ],
)
def test_rl_validation_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        RLConfig(**kwargs).validate()


def test_ui_validation_rejects_unknown_theme():
    with pytest.raises(ValueError, match="theme"):
        UIConfig(theme="rainbow").validate()


def test_round_trip_through_dict():
    """to_dict 之後再 from_dict，內容要完全一樣。"""
    original = SystemConfig()
    original.rl.learning_rate = 0.001
    original.ui.theme = "dark"
    original.traffic.num_intersections = 25

    restored = SystemConfig.from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()
    assert restored.rl.learning_rate == 0.001
    assert restored.ui.theme == "dark"


def test_from_dict_ignores_unknown_keys():
    """設定檔裡多出來的鍵應該被忽略，而不是讓程式炸掉。"""
    data = SystemConfig().to_dict()
    data["這是什麼"] = 123
    data["rl"]["未知參數"] = "abc"

    config = SystemConfig.from_dict(data)
    assert config.rl.learning_rate == RLConfig().learning_rate


def test_from_dict_fills_missing_sections():
    """缺少的區塊要自動補上預設值。"""
    config = SystemConfig.from_dict({"rl": {"batch_size": 64}})

    assert config.rl.batch_size == 64
    assert config.ui.theme == UIConfig().theme


def test_save_and_load(tmp_path):
    path = tmp_path / "nested" / "config.json"

    config = SystemConfig()
    config.ui.chart_height = 500
    saved_path = config.save(path)

    assert saved_path.exists()
    loaded = SystemConfig.load(path)
    assert loaded.ui.chart_height == 500


def test_load_missing_file_returns_defaults(tmp_path):
    """檔案不存在時回傳預設值，不應該丟出例外。"""
    loaded = SystemConfig.load(tmp_path / "does_not_exist.json")
    assert loaded.to_dict() == SystemConfig().to_dict()


def test_load_broken_json_raises(tmp_path):
    """但檔案存在卻是壞的，就要誠實報錯，不要安靜吞掉。"""
    path = tmp_path / "broken.json"
    path.write_text("{ 這不是 JSON", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        SystemConfig.load(path)


def test_api_keys_come_from_environment(monkeypatch):
    """金鑰只從環境變數讀，而且不會出現在 to_dict() 裡。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    config = SystemConfig()

    assert config.openai_api_key == "sk-test-123"
    assert config.api_key_status()["OPENAI_API_KEY"] is True
    assert "sk-test-123" not in json.dumps(config.to_dict())


def test_ensure_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = SystemConfig()
    config.ensure_directories()

    for name in (config.data_dir, config.model_dir, config.log_dir, config.config_dir):
        assert (tmp_path / name).is_dir()


@pytest.mark.parametrize("payload", [[1, 2, 3], "abc", None, 42, 3.5])
def test_from_dict_rejects_non_object_root(payload):
    """`[1,2]`、`"abc"`、`null` 都是合法 JSON 但不是物件。

    沒擋下來的話會在 data.get() 丟出看不懂的 AttributeError，
    在網頁介面上就是整頁當掉。
    """
    with pytest.raises(TypeError, match="最外層必須是物件"):
        SystemConfig.from_dict(payload)


def test_load_non_object_root_raises_typeerror(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(TypeError):
        SystemConfig.load(path)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, 2.0])
def test_rl_validation_rejects_bad_dropout(bad):
    """RLConfig.dropout 會直接傳給 F.dropout。

    之前只有 MultimodalConfig 檢查 dropout，RLConfig 沒有，
    所以 dropout=2 會通過驗證，一路到第一次 GNN forward 才爆。
    """
    with pytest.raises(ValueError, match="dropout"):
        RLConfig(dropout=bad).validate()


def test_rl_validation_accepts_valid_dropout():
    for good in (0.0, 0.1, 0.5, 0.99):
        RLConfig(dropout=good).validate()


@pytest.mark.parametrize("kwargs", [{"hidden_dim": 0}, {"gnn_output_dim": -1}])
def test_rl_validation_rejects_bad_layer_sizes(kwargs):
    with pytest.raises(ValueError, match="hidden_dim|gnn_output_dim"):
        RLConfig(**kwargs).validate()


def test_load_env_file_reads_dotenv(tmp_path, monkeypatch):
    """README 與設定頁都說可以用 .env，所以一定要有人真的去讀它。"""
    pytest.importorskip("dotenv", reason="需要安裝 python-dotenv")

    env = tmp_path / ".env"
    env.write_text("WEATHER_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("WEATHER_API_KEY", raising=False)

    assert load_env_file(env) is True
    assert SystemConfig().weather_api_key == "from-dotenv"


def test_load_env_file_does_not_override_existing(tmp_path, monkeypatch):
    """已經 export 的環境變數優先，不該被 .env 蓋掉。"""
    pytest.importorskip("dotenv", reason="需要安裝 python-dotenv")

    env = tmp_path / ".env"
    env.write_text("WEATHER_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("WEATHER_API_KEY", "from-shell")

    load_env_file(env)
    assert SystemConfig().weather_api_key == "from-shell"


def test_load_env_file_missing_is_not_an_error(tmp_path):
    assert load_env_file(tmp_path / "nope.env") is False


@pytest.mark.parametrize("section_value", [[1], [], "abc", 3, None])
def test_from_dict_rejects_non_object_section(section_value):
    """`{"rl": [1]}` 之類的內容以前會在 .items() 丟出 AttributeError。

    `{"rl": []}` 更糟：因為是 falsey，被 `or {}` 當成「沒給」而安靜忽略。
    """
    with pytest.raises(TypeError, match="rl 區塊必須是物件"):
        SystemConfig.from_dict({"rl": section_value})


def test_from_dict_missing_section_uses_defaults():
    """沒給的區塊還是要用預設值，不能跟著一起被擋下來。"""
    config = SystemConfig.from_dict({"ui": {"chart_height": 500}})
    assert config.rl.batch_size == RLConfig().batch_size
    assert config.ui.chart_height == 500
