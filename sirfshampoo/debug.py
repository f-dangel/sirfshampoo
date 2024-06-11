import math

import torch


def get_H_values(key, G, precond_B_blocks, damping, scaling=1.0):
    results = {}

    # we assume that G = torch.squeeze(p_grad)
    if len(G.shape) == 1:
        name = "%s_dim-%d" % (key, 1)
        B = precond_B_blocks[name]
        d1 = B.shape[0]

        results[name] = torch.zeros(d1, d1, dtype=B.dtype, device=B.device)

        results[name] += (damping / 2.0 * B.t()) @ B
        half = (G.view(1, -1)) @ precond_B_blocks[name]
        results[name].add_(half.t() @ half, alpha=1.0 / 2.0)

        k = torch.tensor(range(d1))
        results[name][k, k] = torch.diagonal(results[name]) - 1.0 / (2.0 * scaling)

    elif len(G.shape) == 2:
        name1 = "%s_dim-%d" % (key, 1)
        name2 = "%s_dim-%d" % (key, 2)
        d1 = precond_B_blocks[name1].shape[0]
        d2 = precond_B_blocks[name2].shape[0]

        B1 = precond_B_blocks[name1] / math.sqrt(d1)
        B2 = precond_B_blocks[name2] / math.sqrt(d2)

        results[name1] = torch.zeros(d1, d1, dtype=B1.dtype, device=B1.device)
        results[name2] = torch.zeros(d2, d2, dtype=B2.dtype, device=B2.device)

        results[name1] = B1.t() @ B1
        results[name2] = B2.t() @ B2

        tr_BBt1 = torch.trace(results[name1])
        tr_BBt2 = torch.trace(results[name2])
        results[name1].mul_((damping * d1) * tr_BBt2 / 2.0)
        results[name2].mul_((damping * d2) * tr_BBt1 / 2.0)

        tmp = B2.t() @ G @ B1
        results[name1].add_(tmp.t() @ tmp, alpha=d1 / 2.0)
        results[name2].add_(tmp @ tmp.t(), alpha=d2 / 2.0)

        k = torch.tensor(range(d1))
        results[name1][k, k] = torch.diagonal(results[name1]) - 1.0 / (2.0 * scaling)

        k = torch.tensor(range(d2))
        results[name2][k, k] = torch.diagonal(results[name2]) - 1.0 / (2.0 * scaling)

    elif len(G.shape) == 3:  # 3d tensor
        name1 = "%s_dim-%d" % (key, 1)
        name2 = "%s_dim-%d" % (key, 2)
        name3 = "%s_dim-%d" % (key, 3)

        d1 = precond_B_blocks[name1].shape[0]
        d2 = precond_B_blocks[name2].shape[0]
        d3 = precond_B_blocks[name3].shape[0]

        B1 = precond_B_blocks[name1] / math.sqrt(d1)
        B2 = precond_B_blocks[name2] / math.sqrt(d2)
        B3 = precond_B_blocks[name3] / math.sqrt(d3)
        results[name1] = B1.t() @ B1
        results[name2] = B2.t() @ B2
        results[name3] = B3.t() @ B3
        tr_BBt1 = torch.trace(results[name1])
        tr_BBt2 = torch.trace(results[name2])
        tr_BBt3 = torch.trace(results[name3])
        results[name1].mul_((d1 * damping) * tr_BBt2 * tr_BBt3 / 2.0)
        results[name2].mul_((d2 * damping) * tr_BBt1 * tr_BBt3 / 2.0)
        results[name3].mul_((d3 * damping) * tr_BBt1 * tr_BBt2 / 2.0)

        tmp_common = torch.einsum("pi,ijk->pjk", B3.t(), G)
        tmp1_half = torch.einsum("pjk,jq->pqk", tmp_common, B2)
        tmp11 = torch.einsum("pqk,ku->pqu", tmp1_half, precond_B_blocks[name1]).view(
            -1, d1
        )
        results[name1].add_(tmp11.t() @ tmp11, alpha=1.0 / 2.0)
        k = torch.tensor(range(d1))
        results[name1][k, k] = torch.diagonal(results[name1]) - 1.0 / (2.0 * scaling)

        tmp2_half = torch.einsum("pjk,km->pjm", tmp_common, B1)
        tmp22 = torch.einsum("pjm,ju->pmu", tmp2_half, precond_B_blocks[name2]).view(
            -1, d2
        )
        results[name2].add_(tmp22.t() @ tmp22, alpha=1.0 / 2.0)

        k = torch.tensor(range(d2))
        results[name2][k, k] = torch.diagonal(results[name2]) - 1.0 / (2.0 * scaling)

        tmp_remaining = torch.einsum("ijk,jq->iqk", G, B2)
        tmp3_half = torch.einsum("iqk,km->iqm", tmp_remaining, B1)
        tmp33 = torch.einsum("iqm,iu->qmu", tmp3_half, precond_B_blocks[name3]).view(
            -1, d3
        )
        results[name3].add_(tmp33.t() @ tmp33, alpha=1.0 / 2.0)
        k = torch.tensor(range(d3))
        results[name3][k, k] = torch.diagonal(results[name3]) - 1.0 / (2.0 * scaling)

    # elif len(G.shape) == 4:  # 4d tensor
    #     name1 = "%s_dim-%d" % (key, 1)
    #     name2 = "%s_dim-%d" % (key, 2)
    #     name3 = "%s_dim-%d" % (key, 3)
    #     name4 = "%s_dim-%d" % (key, 4)
    #     d1 = precond_B_blocks[name1].shape[0]
    #     d2 = precond_B_blocks[name2].shape[0]
    #     d3 = precond_B_blocks[name3].shape[0]
    #     d4 = precond_B_blocks[name4].shape[0]

    #     B1 = precond_B_blocks[name1] / math.sqrt(d1)
    #     B2 = precond_B_blocks[name2] / math.sqrt(d2)
    #     B3 = precond_B_blocks[name3] / math.sqrt(d3)
    #     B4 = precond_B_blocks[name4] / math.sqrt(d4)
    #     results[name1] = B1.t() @ B1
    #     results[name2] = B2.t() @ B2
    #     results[name3] = B3.t() @ B3
    #     results[name4] = B4.t() @ B4
    #     tr_BBt1 = torch.trace(results[name1])
    #     tr_BBt2 = torch.trace(results[name2])
    #     tr_BBt3 = torch.trace(results[name3])
    #     tr_BBt4 = torch.trace(results[name4])
    #     results[name1].mul_((d1 * damping) * tr_BBt2 * tr_BBt3 * tr_BBt4 / 2.0)
    #     results[name2].mul_((d2 * damping) * tr_BBt1 * tr_BBt3 * tr_BBt4 / 2.0)
    #     results[name3].mul_((d3 * damping) * tr_BBt1 * tr_BBt2 * tr_BBt4 / 2.0)
    #     results[name4].mul_((d4 * damping) * tr_BBt1 * tr_BBt2 * tr_BBt3 / 2.0)

    #     tmp_common = torch.einsum("pi,ijkl->pjkl", B4.t(), G)
    #     tmp_a = torch.einsum("pjkl,jq->pqkl", tmp_common, B3)
    #     tmp1_half = torch.einsum("pqkl,km->pqml", tmp_a, B2)
    #     tmp11 = torch.einsum("pqml,lu->pqmu", tmp1_half, precond_B_blocks[name1]).view(
    #         -1, d1
    #     )
    #     results[name1].add_(tmp11.t() @ tmp11, alpha=1.0 / 2.0)
    #     k = torch.tensor(range(d1))
    #     results[name1][k, k] = torch.diagonal(results[name1]) - 1.0 / (2.0 * scaling)

    #     tmp2_half = torch.einsum("pqkl,lw->pqkw", tmp_a, B1)
    #     tmp22 = torch.einsum("pqkw,ku->pqwu", tmp2_half, precond_B_blocks[name2]).view(
    #         -1, d2
    #     )
    #     results[name2].add_(tmp22.t() @ tmp22, alpha=1.0 / 2.0)
    #     k = torch.tensor(range(d2))
    #     results[name2][k, k] = torch.diagonal(results[name2]) - 1.0 / (2.0 * scaling)

    #     tmp_b = torch.einsum("pjkl,km->pjml", tmp_common, B2)
    #     tmp3_half = torch.einsum("pjml,lw->pjmw", tmp_b, B1)
    #     tmp33 = torch.einsum("pjmw,ju->pmwu", tmp3_half, precond_B_blocks[name3]).view(
    #         -1, d3
    #     )
    #     results[name3].add_(tmp33.t() @ tmp33, alpha=1.0 / 2.0)
    #     k = torch.tensor(range(d3))
    #     results[name3][k, k] = torch.diagonal(results[name3]) - 1.0 / (2.0 * scaling)

    #     tmp_remaining = torch.einsum("ijkl,jq->iqkl", G, B3)
    #     tmp_c = torch.einsum("iqkl,km->iqml", tmp_remaining, B2)
    #     tmp4_half = torch.einsum("iqml,lw->iqmw", tmp_c, B1)
    #     tmp44 = torch.einsum("iqmw,iu->qmwu", tmp4_half, precond_B_blocks[name4]).view(
    #         -1, d4
    #     )
    #     results[name4].add_(tmp44.t() @ tmp44, alpha=1.0 / 2.0)
    #     k = torch.tensor(range(d4))
    #     results[name4][k, k] = torch.diagonal(results[name4]) - 1.0 / (2.0 * scaling)

    else:
        raise NotImplementedError

    return results
