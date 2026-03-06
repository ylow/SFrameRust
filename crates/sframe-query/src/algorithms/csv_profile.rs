//! Micro-profiling for CSV read phases.
//! Run with: cargo test -p sframe-query --lib algorithms::csv_profile -- --nocapture --ignored

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::time::Instant;

    use super::super::csv_parallel_tokenizer::{
        ByteConfig, find_true_newline_positions, parallel_tokenize_eof,
        parallel_tokenize_and_parse, split_fields_bytes,
    };
    use super::super::csv_tokenizer::CsvConfig;
    use super::super::csv_parser::parse_cell;
    use sframe_types::flex_type::{FlexType, FlexTypeEnum};
    use std::collections::HashSet;

    fn generate_csv(n: usize) -> Vec<u8> {
        let mut buf = Vec::with_capacity(n * 60);
        buf.extend_from_slice(b"id,category,value,score,label\n");
        for i in 0..n {
            writeln!(
                buf,
                "{},cat_{},{:.2},{},label_{}",
                i,
                i % 100,
                i as f64 * 0.1,
                i % 1000,
                i % 10,
            )
            .unwrap();
        }
        buf
    }

    #[test]
    #[ignore]
    fn profile_csv_phases() {
        let n = 1_000_000;
        eprintln!("Generating {n} rows...");
        let data = generate_csv(n);
        eprintln!("CSV size: {:.1} MB", data.len() as f64 / 1e6);

        let config = CsvConfig::default();
        let bcfg = ByteConfig::from_config(&config).unwrap();

        // Phase 1: Quote parity scan
        let t = Instant::now();
        for _ in 0..5 {
            let _bp = find_true_newline_positions(&data, &bcfg);
        }
        let parity_time = t.elapsed() / 5;
        eprintln!("Quote parity scan:   {:?} ({:.0} MB/s)",
            parity_time, data.len() as f64 / parity_time.as_secs_f64() / 1e6);

        // Phase 2: Parallel tokenize only (to Vec<Vec<String>>)
        let t = Instant::now();
        for _ in 0..5 {
            let (_rows, _lp) = parallel_tokenize_eof(&data, &bcfg);
        }
        let tokenize_time = t.elapsed() / 5;
        eprintln!("Parallel tokenize:   {:?} ({:.0} rows/s)",
            tokenize_time, n as f64 / tokenize_time.as_secs_f64());

        // Phase 2b: Sequential tokenize (single thread split_fields_bytes)
        // to measure per-line cost
        let bp = find_true_newline_positions(&data, &bcfg);
        let _ = bp; // just to build it
        let t = Instant::now();
        let mut count = 0usize;
        let mut pos = 0;
        while pos < data.len() {
            if let Some(nl) = data[pos..].iter().position(|&b| b == b'\n') {
                let line = &data[pos..pos + nl];
                let _fields = split_fields_bytes(line, &bcfg);
                count += 1;
                pos += nl + 1;
            } else {
                break;
            }
        }
        let seq_tokenize_time = t.elapsed();
        eprintln!("Sequential tokenize: {:?} ({:.0} rows/s, {} rows)",
            seq_tokenize_time, count as f64 / seq_tokenize_time.as_secs_f64(), count);

        // Phase 3: Fused tokenize+parse
        let col_types = vec![
            FlexTypeEnum::Integer, // id
            FlexTypeEnum::String,  // category
            FlexTypeEnum::Float,   // value
            FlexTypeEnum::Integer, // score
            FlexTypeEnum::String,  // label
        ];
        let col_indices: Vec<usize> = (0..5).collect();
        let na_values: HashSet<String> = HashSet::new();

        let t = Instant::now();
        for _ in 0..5 {
            let (_cols, _lp) = parallel_tokenize_and_parse(
                &data,
                &bcfg,
                true,
                &col_types,
                &col_indices,
                |val, dtype| {
                    parse_cell(val, dtype, &na_values).unwrap_or(FlexType::Undefined)
                },
            );
        }
        let fused_time = t.elapsed() / 5;
        eprintln!("Fused tokenize+parse: {:?} ({:.0} rows/s)",
            fused_time, n as f64 / fused_time.as_secs_f64());

        // Phase 4: parse_cell only (measure parse overhead)
        let (_rows, _) = parallel_tokenize_eof(&data, &bcfg);
        let rows = _rows;
        let t = Instant::now();
        let mut parse_count = 0usize;
        for row in &rows[1..] {  // skip header
            for (i, field) in row.iter().enumerate() {
                if i < col_types.len() {
                    let _ = parse_cell(field, col_types[i], &na_values);
                    parse_count += 1;
                }
            }
        }
        let parse_time = t.elapsed();
        eprintln!("Sequential parse_cell: {:?} ({:.0} fields/s, {} fields)",
            parse_time, parse_count as f64 / parse_time.as_secs_f64(), parse_count);

        eprintln!("\nSummary:");
        eprintln!("  Parity scan:    {parity_time:?}");
        eprintln!("  Par tokenize:   {tokenize_time:?}");
        eprintln!("  Seq tokenize:   {seq_tokenize_time:?}");
        eprintln!("  Fused tok+parse:{fused_time:?}");
        eprintln!("  Seq parse_cell: {parse_time:?}");
    }
}
