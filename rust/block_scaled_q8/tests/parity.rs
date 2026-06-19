//! RK1 parity gate: the Rust kernel must match the torch reference on a fixture
//! dumped by `scripts/dump_block_scaled_q8_fixture.py`. Reads raw little-endian
//! binaries (no parser dependency, footprint-minimal).
use block_scaled_q8::block_scaled_q8_matmul;
use std::fs;
use std::path::PathBuf;

fn read_f32(p: &PathBuf) -> Vec<f32> {
    fs::read(p)
        .unwrap_or_else(|_| panic!("missing fixture {p:?}; run scripts/dump_block_scaled_q8_fixture.py"))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i8(p: &PathBuf) -> Vec<i8> {
    fs::read(p).unwrap().into_iter().map(|x| x as i8).collect()
}

#[test]
fn parity_with_torch_reference() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/whisper_proj");
    let shape = fs::read_to_string(dir.join("shape.txt"))
        .expect("missing shape.txt; run scripts/dump_block_scaled_q8_fixture.py");
    let v: Vec<usize> = shape.split_whitespace().map(|s| s.parse().unwrap()).collect();
    let (b, m, k, n, bs) = (v[0], v[1], v[2], v[3], v[4]);

    let a = read_f32(&dir.join("a.bin"));
    let qs = read_i8(&dir.join("qs.bin"));
    let d = read_f32(&dir.join("d.bin"));
    let out_ref = read_f32(&dir.join("out.bin"));

    let mut out = vec![0f32; b * m * n];
    block_scaled_q8_matmul(&a, b, m, k, &qs, &d, n, bs, &mut out);

    let mut max_abs = 0f32;
    let (mut num, mut den) = (0f64, 0f64);
    for i in 0..out.len() {
        let e = (out[i] - out_ref[i]).abs();
        max_abs = max_abs.max(e);
        num += (e as f64).powi(2);
        den += (out_ref[i] as f64).powi(2);
    }
    let rel = num.sqrt() / den.sqrt().max(1e-12);
    eprintln!(
        "parity: shape B={b} M={m} K={k} N={n} BS={bs} | max_abs={max_abs:.3e} rel_l2={rel:.3e} ({} elems)",
        out.len()
    );
    assert!(rel < 1e-3, "rel L2 {rel:.3e} exceeds 1e-3 (Rust kernel diverges from torch ref)");
}
