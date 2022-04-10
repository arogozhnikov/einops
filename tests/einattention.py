from einops.einattention import EinAttention
import torch
from einops import rearrange


def minimal_test():
    recipe = EinAttention('t s * head <- t (head *) s, t (head *) s2', head=2)
    result = recipe.forward(torch.ones(3, 4, 5), torch.ones(3, 4, 6), torch.ones(3, 8, 6))
    print(result.shape)
    print(result)

    recipe_jit = torch.jit.script(recipe)
    result2 = recipe_jit(torch.ones(3, 4, 5), torch.ones(3, 4, 6), torch.ones(3, 8, 6))
    assert torch.allclose(result, result2)

    print('jitted version matches')


def test_equivalence_in_different_forms():
    t = 2
    head = 4
    l1 = 2
    r1 = 2
    r2 = 3
    star_qk = 5
    star_v = 7

    q = torch.randn([t, head, l1, star_qk])
    k = torch.randn([t, head, r1, r2, star_qk])
    v = torch.randn([t, head, r1, r2, star_v])

    equivalent_patterns = [
        ('t l1 * head <- t (head *) l1, t (head *) r1 r2', dict(head=head)),
        ('t l1 * head <- t (head *) l1, t (head * r1 r2)', dict(head=head, r1=r1, r2=r2)),
        ('t l1 * head <- t (head * l1), t (head *) r1 r2', dict(head=head, l1=l1)),
        ('t l1 * head <- (t head * l1), (t head * r1 r2)', dict(head=head, t=t, l1=l1, r1=r1, r2=r2)),

        ('(t l1 * head) <- t head * l1, t head * r1 r2', dict()),
        ('(t l1) () head * <- t head * l1, t head * r1 r2', dict()),

        ('(t) l1 * (head) () <- () (t) (head *) l1, () () (t head *) r1 r2', dict(head=head, t=t)),
    ]
    einattention = EinAttention('t head l1 * <- t head l1 *, t head r1 r2 *')
    reference_result = einattention.forward(q=q, k=k, v=v)

    # check that all parameters matter
    assert not torch.allclose(einattention.forward(q * 0, k, v), reference_result)
    assert not torch.allclose(einattention.forward(q, k * 0, v), reference_result)
    assert not torch.allclose(einattention.forward(q, k, v * 0), reference_result)

    for pattern, axis_lengths in equivalent_patterns:
        result_pattern, right = pattern.split('<-')
        q_pattern, kv_pattern = right.split(',')
        result_pattern = result_pattern.replace('*', 'star_v')

        attention = EinAttention(pattern, **axis_lengths)
        result = attention.forward(
            q=rearrange(q, 't head l1 star_qk -> ' + q_pattern.replace('*', 'star_qk')),
            k=rearrange(k, 't head r1 r2 star_qk -> ' + kv_pattern.replace('*', 'star_qk')),
            v=rearrange(v, 't head r1 r2 star_v -> ' + kv_pattern.replace('*', 'star_v')),
        )
        expected_result = rearrange(reference_result, 't head l1 star_v -> ' + result_pattern)
        assert torch.allclose(expected_result, result)


test_equivalence_in_different_forms()

# TODO add tests on wrong number of dimensions
