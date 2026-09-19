"""多模態融合網路的單元測試。

完整的 MultimodalFusionNetwork 需要下載 HuggingFace 與 ImageNet 權重，
在 CI 上不適合跑，所以這裡專注測試 AttentionFusion ──
舊版廣播錯誤的 bug 就在這個類別裡。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="需要安裝 PyTorch")

from multimodal.data_fusion import AttentionFusion  # noqa: E402

EMBED_DIM = 64


@pytest.fixture
def fusion():
    torch.manual_seed(0)
    return AttentionFusion(EMBED_DIM, num_heads=4, dropout=0.0)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 16])
def test_fusion_works_for_any_batch_size(fusion, batch_size):
    """關鍵回歸測試。

    舊版的 `attended_emb * weights.unsqueeze(-1)` 是 [3,B,D] * [B,3,1]，
    只有在 B == 3 時才不會報錯（而且結果還是錯的）。
    現在任何 batch size 都要能正確執行。
    """
    text = torch.randn(batch_size, EMBED_DIM)
    image = torch.randn(batch_size, EMBED_DIM)
    sensor = torch.randn(batch_size, EMBED_DIM)

    fused, weights = fusion(text, image, sensor)

    assert fused.shape == (batch_size, EMBED_DIM)
    assert weights.shape == (batch_size, 3)
    assert torch.isfinite(fused).all()


def test_fusion_weights_sum_to_one(fusion):
    text, image, sensor = (torch.randn(4, EMBED_DIM) for _ in range(3))
    _, weights = fusion(text, image, sensor)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert (weights >= 0).all()


def test_each_sample_uses_its_own_weights(fusion):
    """每筆資料要用自己的那組權重，不能拿到別人的。

    舊版的廣播錯誤剛好會把 batch 維度和模態維度對調，
    等於第 i 筆資料被套上第 i 個模態的權重 ── 完全是錯的。
    """
    fusion.eval()
    batch = torch.randn(4, EMBED_DIM)
    others = torch.randn(4, EMBED_DIM)

    full, _ = fusion(batch, others, others)

    # 單獨跑第 2 筆，結果要和整批一起跑時的第 2 筆一樣
    single, _ = fusion(batch[2:3], others[2:3], others[2:3])

    assert torch.allclose(full[2], single[0], atol=1e-5)


def test_fusion_is_differentiable(fusion):
    text = torch.randn(2, EMBED_DIM, requires_grad=True)
    image = torch.randn(2, EMBED_DIM, requires_grad=True)
    sensor = torch.randn(2, EMBED_DIM, requires_grad=True)

    fused, _ = fusion(text, image, sensor)
    fused.sum().backward()

    for tensor in (text, image, sensor):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_fusion_output_responds_to_each_modality(fusion):
    """改動任一個模態，輸出都應該跟著變 ── 代表三個模態都真的有被用到。"""
    fusion.eval()
    torch.manual_seed(1)
    base = [torch.randn(2, EMBED_DIM) for _ in range(3)]
    reference, _ = fusion(*base)

    for i in range(3):
        modified = list(base)
        modified[i] = modified[i] + 5.0
        changed, _ = fusion(*modified)
        assert not torch.allclose(reference, changed, atol=1e-4)
