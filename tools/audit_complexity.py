"""Analytic parameter and MAC audit for DS-MambaGCN (no PyTorch needed).

Counts follow the paper (Sec. 3, Tables 6-7) and the parameterisation of
`mamba_ssm.Mamba` (in_proj, depthwise conv1d, x_proj, dt_proj, A_log, D,
out_proj). The defaults are the NTU configuration of the paper (Table 6).

    python tools/audit_complexity.py --D 256 --T 300 --expand 2
    python tools/audit_complexity.py --D 256 --T 300 --expand 1 --share-bidir

MACs are multiply-accumulates; most papers (and thop/fvcore) report MACs
under the name "FLOPs". The selective scan is counted explicitly because
thop and fvcore skip the custom CUDA kernel of mamba_ssm.
"""
import argparse
import math


def mamba_params(d, expand, N, d_conv, dt_rank):
    di = expand * d
    return {
        "in_proj": d * 2 * di,
        "conv1d": di * d_conv + di,
        "x_proj": di * (dt_rank + 2 * N),
        "dt_proj": dt_rank * di + di,
        "A_log": di * N,
        "D": di,
        "out_proj": di * d,
    }


def mamba_macs_per_token(d, expand, N, d_conv, dt_rank):
    di = expand * d
    return (d * 2 * di            # in_proj
            + di * d_conv          # depthwise conv
            + di * (dt_rank + 2 * N)  # x_proj
            + dt_rank * di         # dt_proj
            + 3 * di * N           # discretise A, B and state update
            + di * N               # read-out C h
            + di * d)              # out_proj


def audit(a):
    d, T, V, N, L, K = a.D, a.T, a.V, a.N, a.layers, a.classes
    dt_rank = a.dt_rank or math.ceil(d / 16)
    tok = T * V
    P, M = {}, {}

    # Embedding (Eqs. 6-8)
    P["embedding"] = 2 * a.C * d + d + V * d + T * d + 2 * d
    M["embedding"] = tok * 2 * a.C * d

    # Temporal Mamba Stream (Eqs. 16-19): L bidirectional blocks
    mp = sum(mamba_params(d, a.expand, N, a.d_conv, dt_rank).values())
    n_ssm = 1 if a.share_bidir else 2
    block_p = n_ssm * mp + (2 * d * d + d) + 2 * d   # 2 SSMs + merge Linear + LN
    P["temporal_mamba"] = L * block_p
    mm = mamba_macs_per_token(d, a.expand, N, a.d_conv, dt_rank)
    M["temporal_mamba"] = L * tok * (2 * mm + 2 * d * d)  # both directions run

    # Spatial GCN Stream with AGTL (Eqs. 20-23)
    dk = a.dk or d // 4
    gcn_layer_p = d * d + d + 2 * d                   # W, bias, norm
    P["spatial_gcn"] = L * gcn_layer_p + V * V + 2 * (d * dk + dk)  # + A_data + phi_q, phi_k
    M["spatial_gcn"] = L * T * (V * V * d + V * d * d) + V * d * dk * 2 + V * V * dk

    # CSAF (Eqs. 24-27)
    h = a.gate_hidden or d // 4
    P["csaf"] = (4 * d * d + 4 * d) + 2 * 2 * d + (2 * d * h + h + h + 1)
    M["csaf"] = tok * 4 * d * d + 2 * T * V * V * d + 2 * d * h + h

    # Classifier
    P["classifier"] = d * K + K
    M["classifier"] = d * K

    return P, M, dt_rank


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--D", type=int, default=256)
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--V", type=int, default=25)
    p.add_argument("--C", type=int, default=3)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--classes", type=int, default=60)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--d-conv", dest="d_conv", type=int, default=4)
    p.add_argument("--dt-rank", dest="dt_rank", type=int, default=0)
    p.add_argument("--dk", type=int, default=0)
    p.add_argument("--gate-hidden", dest="gate_hidden", type=int, default=0)
    p.add_argument("--share-bidir", dest="share_bidir", action="store_true",
                   help="forward and backward scans share one set of SSM weights")
    a = p.parse_args()
    P, M, r = audit(a)
    print(f"D={a.D} T={a.T} V={a.V} N={a.N} blocks={a.layers} expand={a.expand} "
          f"dt_rank={r} shared_bidir={a.share_bidir}")
    print(f"{'module':<16}{'params (M)':>12}{'GMACs':>10}")
    for k in P:
        print(f"{k:<16}{P[k]/1e6:>12.3f}{M[k]/1e9:>10.2f}")
    print(f"{'TOTAL':<16}{sum(P.values())/1e6:>12.3f}{sum(M.values())/1e9:>10.2f}")


if __name__ == "__main__":
    main()
