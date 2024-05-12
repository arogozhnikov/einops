import pytest

from . import is_backend_tested


def test_torch_einsplit():
    if not is_backend_tested("torch"):
        pytest.skip()

    import torch
    from einops.experimental.einsplit import EinSplit
    b = 2
    s = 3
    c1 = 5
    c2 = 7
    c_out1 = 9
    c_out2 = 11
    mod = EinSplit(f'b s {c1=} {c2=}')
    out1_idx = mod.add_output(f'b s {c_out1}', init='xavier_normal')
    out2_idx = mod.add_output(f'b s {c_out2}', init='xavier_normal')
    out3_idx = mod.add_output(f'(3 b 7) s {c_out2}', init='zeros')
    assert (out1_idx, out2_idx, out3_idx) == (0, 1, 2)

    optim = torch.optim.Adam(mod.parameters(), lr=1e-2)
    batch = torch.randn(b, s, c1, c2)
    out1_norms = []
    out2_norms = []
    out3_norms = []
    for iteration in range(100):
        out1, out2, out3 = mod(batch)
        loss = out1.norm() + out2.norm()
        loss.backward()
        optim.step()
        optim.zero_grad()
        out1_norms.append(out1.norm().item())
        out2_norms.append(out2.norm().item())
        out3_norms.append(out3.norm().item())

        if iteration % 10 == 0:
            print(f'{iteration:>5} {loss:6.2f}')

    assert out3_norms[0] == out3_norms[-1] == 0
    assert out1_norms[0] > 2 * out1_norms[-1]
    assert out2_norms[0] > 2 * out2_norms[-1]