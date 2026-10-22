// Baseline: the fastcdc crate (v2020 module), throughput and chunk statistics.
// usage: fastcdc_bench <min> <avg> <max> files...
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (min, avg, max): (u32, u32, u32) = (a[1].parse().unwrap(), a[2].parse().unwrap(), a[3].parse().unwrap());
    let mut total = 0usize; let mut secs = 0f64; let mut chunks = 0usize; let mut uniq = 0usize;
    let mut seen = std::collections::HashSet::new();
    for f in &a[4..] {
        let data = std::fs::read(f).unwrap();
        let mut best = f64::MAX; let mut ends: Vec<(usize, usize)> = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            let v: Vec<(usize, usize)> = fastcdc::v2020::FastCDC::new(&data, min, avg, max).map(|c| (c.offset, c.length)).collect();
            let d = t.elapsed().as_secs_f64();
            if d < best { best = d; }
            ends = v;
        }
        secs += best; total += data.len(); chunks += ends.len();
        for (o, l) in ends { let h = seahash(&data[o..o + l]); if seen.insert(h) { uniq += l; } }
    }
    println!("fastcdc-rs-v2020,{},{:.3},{},{:.3},{:.4}", total, total as f64 / secs / 1e9, chunks, total as f64 / chunks as f64 / 1024.0, total as f64 / uniq as f64);
}
fn seahash(b: &[u8]) -> u128 { // FNV-1a 64 x2 with different bases: fine for counting
    let (mut h1, mut h2) = (0xcbf29ce484222325u64, 0x84222325cbf29ce4u64);
    for &x in b { h1 = (h1 ^ x as u64).wrapping_mul(0x100000001b3); h2 = (h2 ^ x as u64).wrapping_mul(0x1000193 | (1 << 40)); }
    ((h1 as u128) << 64) | h2 as u128 ^ b.len() as u128
}
