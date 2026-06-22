//! Standalone benchmark binary for the Rust-native block_scaled_q8 kernel (RK2).
//!
//! Generates deterministic inputs for a given (N, K, BS, M) shape — no `rand`
//! crate (footprint) — runs the kernel `ITERS` times and reports the median
//! latency as one JSON line. Peak RSS / binary size are measured externally
//! (`/usr/bin/time -v`, `ls -l`). Usage:
//!
//!     bench N K BS M ITERS
use block_scaled_q8::block_scaled_q8_matmul;
use std::time::Instant;

fn arg(args: &[String], i: usize, default: usize) -> usize {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = arg(&args, 1, 384);
    let k = arg(&args, 2, 384);
    let bs = arg(&args, 3, 32);
    let m = arg(&args, 4, 4);
    let iters = arg(&args, 5, 50);
    let b = 1usize;
    let g = k / bs;

    // Deterministic pseudo-random inputs via a small LCG (no dependency).
    let mut seed = 0x1234_5678u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    };
    let a: Vec<f32> = (0..b * m * k).map(|_| rng()).collect();
    let qs: Vec<i8> = (0..n * g * bs)
        .map(|_| ((rng() * 254.0) as i32).clamp(-127, 127) as i8)
        .collect();
    let d: Vec<f32> = (0..n * g).map(|_| rng().abs() * 0.01 + 1e-3).collect();
    let mut out = vec![0f32; b * m * n];

    // Warm up, then time.
    block_scaled_q8_matmul(&a, b, m, k, &qs, &d, n, bs, &mut out);
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        block_scaled_q8_matmul(&a, b, m, k, &qs, &d, n, bs, &mut out);
        times.push(t.elapsed().as_secs_f64() * 1e3); // ms
    }
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = times[times.len() / 2];
    let checksum: f32 = out.iter().sum(); // prevent dead-code elimination

    println!(
        "{{\"path\":\"rust\",\"n\":{n},\"k\":{k},\"bs\":{bs},\"m\":{m},\"iters\":{iters},\"median_ms\":{median:.6},\"checksum\":{checksum:.6}}}"
    );
}
