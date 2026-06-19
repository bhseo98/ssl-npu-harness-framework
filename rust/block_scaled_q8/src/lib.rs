//! SSLNPU INT8 block-scaled matmul — Rust-native kernel (footprint benchmark twin).
//!
//! Same contract as the IREE-turbine `block_scaled_q8` CustomOp
//! (`src/torch_mlir_zoo/kernels/block_scaled_q8.py`), implemented directly in Rust
//! to compare on-device memory footprint against the MLIR-via-IREE path (RK2).
//!
//! Layout (row-major, contiguous):
//! ```text
//!   a:   [B, M, K]    f32   activation (LHS)
//!   qs:  [N, G, BS]   i8    quantized weight   (G = K / BS)
//!   d:   [N, G]       f32   per-block scale    (the trailing `1` dim collapses)
//!   out: [B, M, N]    f32   = a @ dequant(qs, d)^T
//! ```
//!
//! Dequant is *fused* into the matmul: the weight stays i8 in memory, dequantized
//! per element on the fly (`qs as f32 * scale`) — the full f32 weight is never
//! materialized. Accumulation is f32 (matches the microkernel's accum type).
#![forbid(unsafe_code)]

/// Fused INT8 block-scaled matmul. `out` must be pre-sized to `B*M*N`.
#[allow(clippy::too_many_arguments)]
pub fn block_scaled_q8_matmul(
    a: &[f32],
    b: usize,
    m: usize,
    k: usize,
    qs: &[i8],
    d: &[f32],
    n: usize,
    bs: usize,
    out: &mut [f32],
) {
    let g = k / bs;
    debug_assert_eq!(a.len(), b * m * k);
    debug_assert_eq!(qs.len(), n * g * bs);
    debug_assert_eq!(d.len(), n * g);
    debug_assert_eq!(out.len(), b * m * n);

    for bi in 0..b {
        for mi in 0..m {
            let a_off = (bi * m + mi) * k;
            let out_off = (bi * m + mi) * n;
            for ni in 0..n {
                let mut acc = 0f32;
                for gi in 0..g {
                    let scale = d[ni * g + gi];
                    let qs_off = (ni * g + gi) * bs;
                    let a_blk = a_off + gi * bs;
                    for bsi in 0..bs {
                        // dequant fused: qs(i8) -> f32 * scale; weight stays i8.
                        acc += a[a_blk + bsi] * (qs[qs_off + bsi] as f32 * scale);
                    }
                }
                out[out_off + ni] = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_known_values() {
        // N=1, G=1, BS=2: out = (a0*q0 + a1*q1) * d.
        let a = vec![1.0f32, 2.0];
        let qs = vec![3i8, 4];
        let d = vec![0.5f32];
        let mut out = vec![0f32; 1];
        block_scaled_q8_matmul(&a, 1, 1, 2, &qs, &d, 1, 2, &mut out);
        // (1*3 + 2*4) * 0.5 = 5.5
        assert!((out[0] - 5.5).abs() < 1e-6, "got {}", out[0]);
    }
}
