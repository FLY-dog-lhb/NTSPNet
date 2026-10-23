import torch

from layers import GraphAttention


def test_attention_accepts_per_head_edge_bias_and_preserves_shape():
    layer = GraphAttention(
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        has_edge_attr=True,
        if_self_attention=True,
    ).eval()
    x = torch.randn((3, 8))
    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    edge_attr = torch.randn((2, 8))
    attention_bias = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])

    with torch.no_grad():
        output = layer(
            x,
            edge_index,
            edge_attr,
            attention_bias=attention_bias,
        )

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
