//! Ported from C++ sframe_csv_test.cxx
//!
//! These tests validate that the Rust CSV parser produces identical results
//! to the original C++ SFrame CSV parser across all supported features:
//! delimiters, line endings, quoting, escaping, type inference, NA values,
//! bracket handling, and complex types.

#![cfg(test)]

use std::sync::Arc;

use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use super::csv_parser::{read_csv_string, CsvOptions};

// ---------------------------------------------------------------------------
// Test infrastructure (mirrors C++ csv_test struct and evaluate/validate)
// ---------------------------------------------------------------------------

/// Mirrors the C++ `csv_test` struct: a CSV string, expected column types,
/// expected row values, and tokenizer configuration.
struct CsvTest {
    /// The raw CSV content.
    file: String,
    /// Expected column types in order: (name, type).
    /// When type is Undefined, inference is exercised (the C++ "UNDEFINED" hint).
    types: Vec<(&'static str, FlexTypeEnum)>,
    /// Expected row values. `values[row][col]`.
    values: Vec<Vec<FlexType>>,
    /// Parser options.
    options: CsvOptions,
}

impl CsvTest {
    fn new() -> Self {
        CsvTest {
            file: String::new(),
            types: Vec::new(),
            values: Vec::new(),
            options: CsvOptions::default(),
        }
    }
}

/// Assert two FlexType values are equal, using approximate comparison for
/// floats and recursive comparison for containers.
fn assert_flex_eq(actual: &FlexType, expected: &FlexType, row: usize, col: usize) {
    match (actual, expected) {
        (FlexType::Undefined, FlexType::Undefined) => {}
        (FlexType::Integer(a), FlexType::Integer(b)) => {
            assert_eq!(a, b, "row={row} col={col}: Integer mismatch");
        }
        (FlexType::Float(a), FlexType::Float(b)) => {
            assert!(
                (a - b).abs() < 1e-5,
                "row={row} col={col}: Float mismatch: {a} vs {b}"
            );
        }
        (FlexType::String(a), FlexType::String(b)) => {
            assert_eq!(
                a.as_ref(),
                b.as_ref(),
                "row={row} col={col}: String mismatch"
            );
        }
        (FlexType::Vector(a), FlexType::Vector(b)) => {
            assert_eq!(
                a.len(),
                b.len(),
                "row={row} col={col}: Vector length mismatch"
            );
            for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (va - vb).abs() < 1e-5,
                    "row={row} col={col} elem={i}: Vector element mismatch: {va} vs {vb}"
                );
            }
        }
        (FlexType::List(a), FlexType::List(b)) => {
            assert_eq!(
                a.len(),
                b.len(),
                "row={row} col={col}: List length mismatch"
            );
            for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
                assert_flex_eq(va, vb, row, col);
                let _ = i; // suppress unused
            }
        }
        (FlexType::Dict(a), FlexType::Dict(b)) => {
            assert_eq!(
                a.len(),
                b.len(),
                "row={} col={}: Dict length mismatch: got {} entries, expected {}",
                row,
                col,
                a.len(),
                b.len()
            );
            for (i, ((ka, va), (kb, vb))) in a.iter().zip(b.iter()).enumerate() {
                assert_flex_eq(ka, kb, row, col);
                assert_flex_eq(va, vb, row, col);
                let _ = i;
            }
        }
        _ => {
            panic!(
                "row={row} col={col}: type mismatch: got {actual:?}, expected {expected:?}"
            );
        }
    }
}

/// Parse the CSV and validate against expected types and values.
/// When test.types specifies non-Undefined types, they are passed as type hints
/// to the parser (matching C++ behavior where types serve as both hints and
/// expected output).
fn validate(test: &CsvTest) {
    let mut opts = test.options.clone();
    // Pass ALL types as hints, including Undefined. When Undefined, the column
    // uses Flexible storage with per-value parsing (matching C++ behavior).
    let hints: Vec<(String, FlexTypeEnum)> = test
        .types
        .iter()
        .map(|(name, t)| (name.to_string(), *t))
        .collect();
    opts.type_hints = hints;
    let (col_names, batch) =
        read_csv_string(&test.file, &opts).expect("CSV parse failed");

    // Check column count
    assert_eq!(
        col_names.len(),
        test.types.len(),
        "Column count mismatch: got {:?}, expected {:?}",
        col_names,
        test.types.iter().map(|(n, _)| *n).collect::<Vec<_>>()
    );

    // Check column names and types
    for (i, (expected_name, expected_type)) in test.types.iter().enumerate() {
        assert_eq!(
            col_names[i], *expected_name,
            "Column {i} name mismatch"
        );
        // When expected type is Undefined, we accept whatever was inferred
        if *expected_type != FlexTypeEnum::Undefined {
            assert_eq!(
                batch.dtypes()[i], *expected_type,
                "Column '{}' (idx {}) type mismatch: got {:?}",
                expected_name, i, batch.dtypes()[i]
            );
        }
    }

    // Check row count
    assert_eq!(
        batch.num_rows(),
        test.values.len(),
        "Row count mismatch: got {}, expected {}",
        batch.num_rows(),
        test.values.len()
    );

    // Check values
    for (row_idx, expected_row) in test.values.iter().enumerate() {
        for (col_idx, expected_val) in expected_row.iter().enumerate() {
            let actual = batch.column(col_idx).get(row_idx);
            assert_flex_eq(&actual, expected_val, row_idx, col_idx);
        }
    }
}

// ---------------------------------------------------------------------------
// FlexType construction helpers
// ---------------------------------------------------------------------------

fn int(v: i64) -> FlexType {
    FlexType::Integer(v)
}
fn float(v: f64) -> FlexType {
    FlexType::Float(v)
}
fn string(v: &str) -> FlexType {
    FlexType::String(Arc::from(v))
}
fn vec_f(v: &[f64]) -> FlexType {
    FlexType::Vector(Arc::from(v))
}
fn list(v: Vec<FlexType>) -> FlexType {
    FlexType::List(Arc::from(v))
}
fn dict(pairs: Vec<(FlexType, FlexType)>) -> FlexType {
    FlexType::Dict(Arc::from(pairs))
}
fn undef() -> FlexType {
    FlexType::Undefined
}

// ---------------------------------------------------------------------------
// Test case constructors (mirror C++ functions)
// ---------------------------------------------------------------------------

/// Basic CSV with one of every parseable type.
fn basic(dlm: &str, line_ending: &str) -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = dlm.to_string();
    if line_ending != "\n" && line_ending != "\r\n" && line_ending != "\r" {
        t.options.line_terminator = line_ending.to_string();
    }

    let d = dlm;
    let le = line_ending;
    t.file = format!(
        "float{d}int{d}str{d}vec{d}dict{d}rec{le}\
         1.1{d}1{d}one{d}[1,1,1]{d}{{1:1,\"a\":\"a\"}}{d}[a,a]{le}\
         2.2{d}2{d}two{d}[2,2,2]{d}{{2:2,\"b\":\"b\"}}{d}[b,b]{le}\
         3.3{d}3{d}three{d}[3,3,3]{d}{{3:3,\"c\":\"c\"}}{d}[c,c]{le}"
    );

    t.values = vec![
        vec![
            float(1.1),
            int(1),
            string("one"),
            vec_f(&[1.0, 1.0, 1.0]),
            dict(vec![(int(1), int(1)), (string("a"), string("a"))]),
            list(vec![string("a"), string("a")]),
        ],
        vec![
            float(2.2),
            int(2),
            string("two"),
            vec_f(&[2.0, 2.0, 2.0]),
            dict(vec![(int(2), int(2)), (string("b"), string("b"))]),
            list(vec![string("b"), string("b")]),
        ],
        vec![
            float(3.3),
            int(3),
            string("three"),
            vec_f(&[3.0, 3.0, 3.0]),
            dict(vec![(int(3), int(3)), (string("c"), string("c"))]),
            list(vec![string("c"), string("c")]),
        ],
    ];

    t.types = vec![
        ("float", FlexTypeEnum::Float),
        ("int", FlexTypeEnum::Integer),
        ("str", FlexTypeEnum::String),
        ("vec", FlexTypeEnum::Vector),
        ("dict", FlexTypeEnum::Dict),
        ("rec", FlexTypeEnum::List),
    ];
    t
}

/// Basic CSV with comments and skip_rows.
fn basic_comments_and_skips(dlm: &str, line_ending: &str) -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = dlm.to_string();
    if line_ending != "\n" && line_ending != "\r\n" && line_ending != "\r" {
        t.options.line_terminator = line_ending.to_string();
        t.options.comment_char = Some('#');
    }

    let d = dlm;
    let le = line_ending;
    t.file = format!(
        "junk{le}\
         trash{le}\
         # a commented string{le}\
         float{d}int{d}str{d}vec{d}dict{d}rec{le}\
         1.1{d}1{d}one{d}[1,1,1]{d}{{1:1,\"a\":\"a\"}}{d}[a,a]{le}\
         # another commented string{le}\
         2.2{d}2{d}two{d}[2,2,2]{d}{{2:2,\"b\":\"b\"}}{d}[b,b]{le}\
         3.3{d}3{d}three{d}[3,3,3]{d}{{3:3,\"c\":\"c\"}}{d}[c,c]{le}"
    );
    t.options.skip_rows = 2;

    t.values = vec![
        vec![
            float(1.1),
            int(1),
            string("one"),
            vec_f(&[1.0, 1.0, 1.0]),
            dict(vec![(int(1), int(1)), (string("a"), string("a"))]),
            list(vec![string("a"), string("a")]),
        ],
        vec![
            float(2.2),
            int(2),
            string("two"),
            vec_f(&[2.0, 2.0, 2.0]),
            dict(vec![(int(2), int(2)), (string("b"), string("b"))]),
            list(vec![string("b"), string("b")]),
        ],
        vec![
            float(3.3),
            int(3),
            string("three"),
            vec_f(&[3.0, 3.0, 3.0]),
            dict(vec![(int(3), int(3)), (string("c"), string("c"))]),
            list(vec![string("c"), string("c")]),
        ],
    ];

    t.types = vec![
        ("float", FlexTypeEnum::Float),
        ("int", FlexTypeEnum::Integer),
        ("str", FlexTypeEnum::String),
        ("vec", FlexTypeEnum::Vector),
        ("dict", FlexTypeEnum::Dict),
        ("rec", FlexTypeEnum::List),
    ];
    t
}

/// Escape-string helper that wraps a value in quotes with backslash escaping.
/// Mirrors the C++ `default_escape_string()`.
fn esc(s: &str) -> String {
    let inner = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    format!("\"{inner}\"")
}

/// Basic CSV with all fields quoted.
fn quoted_basic(dlm: &str, line_ending: &str) -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = dlm.to_string();
    if line_ending != "\n" && line_ending != "\r\n" && line_ending != "\r" {
        t.options.line_terminator = line_ending.to_string();
    }

    let d = dlm;
    let le = line_ending;
    t.file = format!(
        "{}{d}{}{d}{}{d}{}{d}{}{d}{}{le}\
         {}{d}{}{d}{}{d}{}{d}{}{d}{}{le}\
         {}{d}{}{d}{}{d}{}{d}{}{d}{}{le}\
         {}{d}{}{d}{}{d}{}{d}{}{d}{}{le}",
        esc("float"),
        esc("int"),
        esc("str"),
        esc("vec"),
        esc("dict"),
        esc("rec"),
        esc("1.1"),
        esc("1"),
        esc("one"),
        esc("[1,1,1]"),
        esc("{1:1,\"a\":\"a\"}"),
        esc("[a,a]"),
        esc("2.2"),
        esc("2"),
        esc("two"),
        esc("[2,2,2]"),
        esc("{2:2,\"b\":\"b\"}"),
        esc("[b,b]"),
        esc("3.3"),
        esc("3"),
        esc("three"),
        esc("[3,3,3]"),
        esc("{3:3,\"c\":\"c\"}"),
        esc("[c,c]"),
    );

    t.values = vec![
        vec![
            float(1.1),
            int(1),
            string("one"),
            vec_f(&[1.0, 1.0, 1.0]),
            dict(vec![(int(1), int(1)), (string("a"), string("a"))]),
            list(vec![string("a"), string("a")]),
        ],
        vec![
            float(2.2),
            int(2),
            string("two"),
            vec_f(&[2.0, 2.0, 2.0]),
            dict(vec![(int(2), int(2)), (string("b"), string("b"))]),
            list(vec![string("b"), string("b")]),
        ],
        vec![
            float(3.3),
            int(3),
            string("three"),
            vec_f(&[3.0, 3.0, 3.0]),
            dict(vec![(int(3), int(3)), (string("c"), string("c"))]),
            list(vec![string("c"), string("c")]),
        ],
    ];

    t.types = vec![
        ("float", FlexTypeEnum::Float),
        ("int", FlexTypeEnum::Integer),
        ("str", FlexTypeEnum::String),
        ("vec", FlexTypeEnum::Vector),
        ("dict", FlexTypeEnum::Dict),
        ("rec", FlexTypeEnum::List),
    ];
    t
}

/// Type inference test — all types set to Undefined (auto-infer).
fn test_type_inference(dlm: &str, line_ending: &str) -> CsvTest {
    let mut t = basic(dlm, line_ending);
    // Set all types to Undefined to trigger inference
    t.types = vec![
        ("float", FlexTypeEnum::Undefined),
        ("int", FlexTypeEnum::Undefined),
        ("str", FlexTypeEnum::Undefined),
        ("vec", FlexTypeEnum::Undefined),
        ("dict", FlexTypeEnum::Undefined),
        ("rec", FlexTypeEnum::Undefined),
    ];
    t
}

/// Type inference with all fields quoted.
fn test_quoted_type_inference(dlm: &str, line_ending: &str) -> CsvTest {
    let mut t = quoted_basic(dlm, line_ending);
    t.types = vec![
        ("float", FlexTypeEnum::Undefined),
        ("int", FlexTypeEnum::Undefined),
        ("str", FlexTypeEnum::Undefined),
        ("vec", FlexTypeEnum::Undefined),
        ("dict", FlexTypeEnum::Undefined),
        ("rec", FlexTypeEnum::Undefined),
    ];
    t
}

/// Embedded bracket characters in string fields alongside vector fields.
fn test_embedded_strings(dlm: &str) -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = dlm.to_string();
    let d = dlm;
    t.file = format!(
        "str{d}vec\n\
         [abc{d}[1,1,1]\n\
         cde]{d}[2,2,2]\n\
         a[a]b{d}[3,3,3]\n\
         \"[abc\"{d}[1,1,1]\n\
         \"cde]\"{d}[2,2,2]\n\
         \"a[a]b\"{d}[3,3,3]\n"
    );

    t.values = vec![
        vec![string("[abc"), vec_f(&[1.0, 1.0, 1.0])],
        vec![string("cde]"), vec_f(&[2.0, 2.0, 2.0])],
        vec![string("a[a]b"), vec_f(&[3.0, 3.0, 3.0])],
        vec![string("[abc"), vec_f(&[1.0, 1.0, 1.0])],
        vec![string("cde]"), vec_f(&[2.0, 2.0, 2.0])],
        vec![string("a[a]b"), vec_f(&[3.0, 3.0, 3.0])],
    ];

    t.types = vec![
        ("str", FlexTypeEnum::String),
        ("vec", FlexTypeEnum::Vector),
    ];
    t
}

/// Quoted embedded bracket characters.
fn test_quoted_embedded_strings(dlm: &str) -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = dlm.to_string();
    let d = dlm;
    t.file = format!(
        "str{d}vec\n\
         {}{d}{}\n\
         {}{d}{}\n\
         {}{d}{}\n\
         {}{d}{}\n\
         {}{d}{}\n\
         {}{d}{}\n",
        esc("[abc"),
        esc("[1,1,1]"),
        esc("cde]"),
        esc("[2,2,2]"),
        esc("a[a]b"),
        esc("[3,3,3]"),
        esc("[abc"),
        esc("[1,1,1]"),
        esc("cde]"),
        esc("[2,2,2]"),
        esc("a[a]b"),
        esc("[3,3,3]"),
    );

    t.values = vec![
        vec![string("[abc"), vec_f(&[1.0, 1.0, 1.0])],
        vec![string("cde]"), vec_f(&[2.0, 2.0, 2.0])],
        vec![string("a[a]b"), vec_f(&[3.0, 3.0, 3.0])],
        vec![string("[abc"), vec_f(&[1.0, 1.0, 1.0])],
        vec![string("cde]"), vec_f(&[2.0, 2.0, 2.0])],
        vec![string("a[a]b"), vec_f(&[3.0, 3.0, 3.0])],
    ];

    t.types = vec![
        ("str", FlexTypeEnum::String),
        ("vec", FlexTypeEnum::Vector),
    ];
    t
}

/// Complex CSV: semicolon delimiter, double-quote, NA values, escaping.
fn interesting() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = ";".to_string();
    t.options.double_quote = true;
    t.options.na_values = vec!["NA".into(), "Pokemon".into(), "".into()];
    t.file = concat!(
        "#this is a comment\n",
        "float;int;vec;str #this is another comment\n",
        "1.1 ;1;[1 2 3];\"hello\\\\\"\n",
        "2.2;2; [4 5 6];\"wor;ld\"\n",
        " 3.3; 3;[9 2];\"\"\"w\"\"\"\n",
        "Pokemon  ;;; NA ",
    )
    .to_string();

    t.values = vec![
        vec![float(1.1), int(1), vec_f(&[1.0, 2.0, 3.0]), string("hello\\")],
        vec![float(2.2), int(2), vec_f(&[4.0, 5.0, 6.0]), string("wor;ld")],
        vec![float(3.3), int(3), vec_f(&[9.0, 2.0]), string("\"w\"")],
        vec![undef(), undef(), undef(), undef()],
    ];

    t.types = vec![
        ("float", FlexTypeEnum::Float),
        ("int", FlexTypeEnum::Integer),
        ("vec", FlexTypeEnum::Vector),
        ("str", FlexTypeEnum::String),
    ];
    t
}

/// Space delimiter with excess whitespace around fields.
fn excess_white_space() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = " ".to_string();
    t.file = concat!(
        "float int str  vec    dict rec\n",
        "  1.1 1 one   [1,1,1]  {1 : 1 , \"a\"  : \"a\"}    [a,a]\n",
        " 2.2 2 two   [2,2,2] {2:2,\"b\":\"b\"} [b,b]\n",
        "3.3 3 three [3,3,3]  {3:3,  \"c\":\"c\"} [c,c]  \t\n",
    )
    .to_string();

    t.values = vec![
        vec![
            float(1.1),
            int(1),
            string("one"),
            vec_f(&[1.0, 1.0, 1.0]),
            dict(vec![(int(1), int(1)), (string("a"), string("a"))]),
            list(vec![string("a"), string("a")]),
        ],
        vec![
            float(2.2),
            int(2),
            string("two"),
            vec_f(&[2.0, 2.0, 2.0]),
            dict(vec![(int(2), int(2)), (string("b"), string("b"))]),
            list(vec![string("b"), string("b")]),
        ],
        vec![
            float(3.3),
            int(3),
            string("three"),
            vec_f(&[3.0, 3.0, 3.0]),
            dict(vec![(int(3), int(3)), (string("c"), string("c"))]),
            list(vec![string("c"), string("c")]),
        ],
    ];

    t.types = vec![
        ("float", FlexTypeEnum::Float),
        ("int", FlexTypeEnum::Integer),
        ("str", FlexTypeEnum::String),
        ("vec", FlexTypeEnum::Vector),
        ("dict", FlexTypeEnum::Dict),
        ("rec", FlexTypeEnum::List),
    ];
    t
}

/// Issue #1514: unmatched brackets/braces as string values.
fn another_wierd_bracketing_thing_issue_1514() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = "\t".to_string();
    t.file = concat!(
        "X1\tX2\tX3\tX4\tX5\tX6\tX7\tX8\tX9\n",
        "1\t{\t()\t{}\t{}\t(}\t})\t}\tdebugging\n",
        "3\t--\t({})\t{()}\t{}\t({\t{)\t}\tdebugging\n",
    )
    .to_string();

    t.values = vec![
        vec![
            string("1"),
            string("{"),
            string("()"),
            string("{}"),
            string("{}"),
            string("(}"),
            string("})"),
            string("}"),
            string("debugging"),
        ],
        vec![
            string("3"),
            string("--"),
            string("({})"),
            string("{()}"),
            string("{}"),
            string("({"),
            string("{)"),
            string("}"),
            string("debugging"),
        ],
    ];

    t.types = vec![
        ("X1", FlexTypeEnum::String),
        ("X2", FlexTypeEnum::String),
        ("X3", FlexTypeEnum::String),
        ("X4", FlexTypeEnum::String),
        ("X5", FlexTypeEnum::String),
        ("X6", FlexTypeEnum::String),
        ("X7", FlexTypeEnum::String),
        ("X8", FlexTypeEnum::String),
        ("X9", FlexTypeEnum::String),
    ];
    t
}

/// NA values test.
fn make_na_values() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.na_values = vec!["NA".into(), "PIKA".into(), "CHU".into()];
    t.file = "a,b,c\nNA,PIKA,CHU\n1.0,2,3\n".to_string();

    t.values = vec![
        vec![undef(), undef(), undef()],
        vec![float(1.0), int(2), int(3)],
    ];

    t.types = vec![
        ("a", FlexTypeEnum::Float),
        ("b", FlexTypeEnum::Integer),
        ("c", FlexTypeEnum::Integer),
    ];
    t
}

/// NA values with "-8" as NA.
fn make_na_values2() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.na_values = vec!["-8".into()];
    t.file = "k,v\na,1\nb,1\nc,-8\nd,3\n".to_string();

    t.values = vec![
        vec![string("a"), int(1)],
        vec![string("b"), int(1)],
        vec![string("c"), undef()],
        vec![string("d"), int(3)],
    ];

    t.types = vec![
        ("k", FlexTypeEnum::String),
        ("v", FlexTypeEnum::Integer),
    ];
    t
}

/// Tab-delimited with missing values.
fn make_missing_tab_values() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = "\t".to_string();
    t.file = "a\tb\tc\n1\t\t  b\n2\t\t\n3\t  c\t d \n".to_string();

    t.values = vec![
        vec![int(1), undef(), string("b")],
        vec![int(2), undef(), undef()],
        vec![int(3), string("c"), string("d")],
    ];

    t.types = vec![
        ("a", FlexTypeEnum::Undefined),
        ("b", FlexTypeEnum::Undefined),
        ("c", FlexTypeEnum::Undefined),
    ];
    t
}

/// Quoted integers with double-quote escaping (content has literal quotes).
fn string_integers() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.double_quote = true;
    // CSV: int,str\n1,"""1"""\n2,"\"2\""
    t.file = "int,str\n1,\"\"\"1\"\"\"\n2,\"\\\"2\\\"\"\n".to_string();

    t.values = vec![
        vec![int(1), string("\"1\"")],
        vec![int(2), string("\"2\"")],
    ];

    t.types = vec![
        ("int", FlexTypeEnum::Undefined),
        ("str", FlexTypeEnum::Undefined),
    ];
    t
}

/// Quoted integers that parse through to integer type.
fn string_integers2() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.double_quote = true;
    t.file = "int,str\n1,\"1\"\n2,\"2\"\n".to_string();

    t.values = vec![vec![int(1), int(1)], vec![int(2), int(2)]];

    t.types = vec![
        ("int", FlexTypeEnum::Undefined),
        ("str", FlexTypeEnum::Undefined),
    ];
    t
}

/// Custom line terminator "zzz".
fn alternate_endline_test() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = " ".to_string();
    t.options.line_terminator = "zzz".to_string();
    t.file = "a b czzz 1 2 3zzz\n".to_string();

    t.values = vec![vec![int(1), int(2), int(3)]];

    t.types = vec![
        ("a", FlexTypeEnum::Undefined),
        ("b", FlexTypeEnum::Undefined),
        ("c", FlexTypeEnum::Undefined),
    ];
    t
}

/// Escape sequences in quoted fields (space delimiter).
fn escape_parsing() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = " ".to_string();
    t.file = concat!(
        "str1 str2\n",
        "\"\\n\"  \"\\n\"\n",
        "\"\\t\"  \"\\0abf\"\n",
        "\"\\\"a\"  \"\\\"b\"\n",
        "{\"a\":\"\\\"\"} [a,\"b\",\"\\\"c\"]\n",
    )
    .to_string();

    t.values = vec![
        vec![string("\n"), string("\n")],
        vec![string("\t"), string("\\0abf")],
        vec![string("\"a"), string("\"b")],
        vec![
            dict(vec![(string("a"), string("\""))]),
            list(vec![string("a"), string("b"), string("\"c")]),
        ],
    ];

    t.types = vec![
        ("str1", FlexTypeEnum::Undefined),
        ("str2", FlexTypeEnum::Undefined),
    ];
    t
}

/// Escape sequences with explicit STRING type hints.
fn escape_parsing_string_hint() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = " ".to_string();
    t.options.type_hints = vec![
        ("str1".to_string(), FlexTypeEnum::String),
        ("str2".to_string(), FlexTypeEnum::String),
    ];
    t.file = "str1 str2\n\"\\n\"  \"\\n\"\n\"\\t\"  \"\\0abf\"\n".to_string();

    t.values = vec![
        vec![string("\n"), string("\n")],
        vec![string("\t"), string("\\0abf")],
    ];

    t.types = vec![
        ("str1", FlexTypeEnum::String),
        ("str2", FlexTypeEnum::String),
    ];
    t
}

/// Unquoted escape sequences stay literal (not interpreted).
fn non_escaped_parsing() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = " ".to_string();
    t.options.type_hints = vec![
        ("str1".to_string(), FlexTypeEnum::String),
        ("str2".to_string(), FlexTypeEnum::String),
    ];
    t.file = "str1 str2\n\\n  \\n\n\\t  \\0abf\n".to_string();

    t.values = vec![
        vec![string("\\n"), string("\\n")],
        vec![string("\\t"), string("\\0abf")],
    ];

    t.types = vec![
        ("str1", FlexTypeEnum::String),
        ("str2", FlexTypeEnum::String),
    ];
    t
}

/// Single string column with newline delimiter.
fn single_string_column() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = "\n".to_string();
    t.options.type_hints = vec![("str1".to_string(), FlexTypeEnum::String)];
    t.file = "str1\n\"\"\n{\"a\":\"b\"}\n{\"\":\"\"}\n".to_string();

    t.values = vec![
        vec![string("")],
        vec![string("{\"a\":\"b\"}")],
        vec![string("{\"\":\"\"}")],
    ];

    t.types = vec![("str1", FlexTypeEnum::String)];
    t
}

/// Unicode surrogate pairs in dict values.
fn unicode_surrogate_pairs() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = "\n".to_string();
    t.options.type_hints = vec![("dict".to_string(), FlexTypeEnum::Dict)];

    let mut file = String::from("dict\n");
    file.push_str("{\"good_surrogates\": \"\\uD834\\uDD1E\"}\n");
    // The C++ test uses a special right-quote character \u2019 after the bad
    // surrogate. We replicate that exactly.
    file.push_str("{\"bad_surrogates\": \"\\uD834\u{2019}\"}\n");
    file.push_str("{\"bad_surrogates2\": \"\\uD834\" }\n");
    file.push_str("{\"bad_surrogates3\": \"\\uD834\\uDD\" }\n");
    file.push_str("{\"bad_json\": \"\\u442G\" }");
    t.file = file;

    t.values = vec![
        vec![dict(vec![(string("good_surrogates"), string("𝄞"))])],
        vec![dict(vec![(
            string("bad_surrogates"),
            string("\\uD834\u{2019}"),
        )])],
        vec![dict(vec![(string("bad_surrogates2"), string("\\uD834"))])],
        vec![dict(vec![(
            string("bad_surrogates3"),
            string("\\uD834\\uDD"),
        )])],
        vec![dict(vec![(string("bad_json"), string("\\u442G"))])],
    ];

    t.types = vec![("dict", FlexTypeEnum::Dict)];
    t
}

/// Multiline JSON with empty delimiter.
fn multiline_json() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = "".to_string();
    t.options.line_terminator = "".to_string();
    t.options.has_header = false;
    t.options.type_hints = vec![("X1".to_string(), FlexTypeEnum::Dict)];
    t.file = "{\n       \"glossary\": 123,\n       \"fish\": 456\n        }".to_string();

    t.values = vec![vec![dict(vec![
        (string("glossary"), int(123)),
        (string("fish"), int(456)),
    ])]];

    t.types = vec![("X1", FlexTypeEnum::Dict)];
    t
}

/// Tab-delimited with list columns, no header.
fn tab_delimited_csv_with_list() -> CsvTest {
    let mut t = CsvTest::new();
    t.options.delimiter = "\t".to_string();
    t.options.has_header = false;
    t.file = "xxx\t[1,2,3]\t[1,2,3]\n".to_string();

    t.values = vec![vec![
        string("xxx"),
        list(vec![int(1), int(2), int(3)]),
        list(vec![int(1), int(2), int(3)]),
    ]];

    t.types = vec![
        ("X1", FlexTypeEnum::String),
        ("X2", FlexTypeEnum::List),
        ("X3", FlexTypeEnum::List),
    ];
    t
}

// ===========================================================================
// Test functions
// ===========================================================================

// ---- basic tests with various delimiters and line endings ----

#[test]
fn test_basic_comma() {
    validate(&basic(",", "\n"));
}

#[test]
fn test_basic_comma_cr() {
    validate(&basic(",", "\r"));
}

#[test]
fn test_basic_comma_crlf() {
    validate(&basic(",", "\r\n"));
}

#[test]
fn test_basic_comma_custom_abc() {
    validate(&basic(",", "abc"));
}

#[test]
fn test_basic_comma_custom_aaaaaa() {
    validate(&basic(",", "aaaaaa"));
}

#[test]
fn test_basic_space() {
    validate(&basic(" ", "\n"));
}

#[test]
fn test_basic_space_cr() {
    validate(&basic(" ", "\r"));
}

#[test]
fn test_basic_space_crlf() {
    validate(&basic(" ", "\r\n"));
}

#[test]
fn test_basic_space_custom_abc() {
    validate(&basic(" ", "abc"));
}

#[test]
fn test_basic_space_custom_bbbbbb() {
    validate(&basic(" ", "bbbbbb"));
}

#[test]
fn test_basic_semicolon() {
    validate(&basic(";", "\n"));
}

#[test]
fn test_basic_semicolon_cr() {
    validate(&basic(";", "\r"));
}

#[test]
fn test_basic_semicolon_crlf() {
    validate(&basic(";", "\r\n"));
}

#[test]
fn test_basic_semicolon_custom_pokemon() {
    validate(&basic(";", "pokemon"));
}

#[test]
fn test_basic_double_colon() {
    validate(&basic("::", "\n"));
}

#[test]
fn test_basic_double_space() {
    validate(&basic("  ", "\n"));
}

#[test]
fn test_basic_double_tab() {
    validate(&basic("\t\t", "\n"));
}

// ---- comments and skips ----

#[test]
fn test_basic_comments_and_skips_comma() {
    validate(&basic_comments_and_skips(",", "\n"));
}

// ---- interesting (complex) ----

#[test]
fn test_interesting() {
    validate(&interesting());
}

// ---- excess whitespace with space delimiter ----

#[test]
fn test_excess_white_space() {
    validate(&excess_white_space());
}

// ---- embedded strings ----

#[test]
fn test_embedded_strings_comma() {
    validate(&test_embedded_strings(","));
}

#[test]
fn test_embedded_strings_space() {
    validate(&test_embedded_strings(" "));
}

#[test]
fn test_embedded_strings_tab() {
    validate(&test_embedded_strings("\t"));
}

#[test]
fn test_embedded_strings_double_tab() {
    validate(&test_embedded_strings("\t\t"));
}

#[test]
fn test_embedded_strings_double_space() {
    validate(&test_embedded_strings("  "));
}

#[test]
fn test_embedded_strings_double_colon() {
    validate(&test_embedded_strings("::"));
}

// ---- issue 1514 ----

#[test]
fn test_issue_1514_bracket_thing() {
    validate(&another_wierd_bracketing_thing_issue_1514());
}

// ---- type inference ----

#[test]
fn test_type_inference_comma() {
    validate(&test_type_inference(",", "\n"));
}

#[test]
fn test_type_inference_comma_custom_zzz() {
    validate(&test_type_inference(",", "zzz"));
}

// ---- string_integers ----

#[test]
fn test_string_integers() {
    validate(&string_integers());
}

#[test]
fn test_string_integers2() {
    validate(&string_integers2());
}

// ---- escape parsing ----

#[test]
fn test_escape_parsing() {
    validate(&escape_parsing());
}

#[test]
fn test_escape_parsing_string_hint() {
    validate(&escape_parsing_string_hint());
}

#[test]
fn test_non_escaped_parsing() {
    validate(&non_escaped_parsing());
}

// ---- single string column ----

#[test]
fn test_single_string_column() {
    validate(&single_string_column());
}

// ---- missing tab values ----

#[test]
fn test_missing_tab_values() {
    validate(&make_missing_tab_values());
}

// ---- tab delimited list ----

#[test]
fn test_tab_delimited_csv_with_list() {
    validate(&tab_delimited_csv_with_list());
}

// ---- NA values ----

#[test]
fn test_na_values() {
    validate(&make_na_values());
}

#[test]
fn test_na_values2() {
    validate(&make_na_values2());
}

// ---- quoted basic ----

#[test]
fn test_quoted_basic_comma() {
    validate(&quoted_basic(",", "\n"));
}

#[test]
fn test_quoted_basic_comma_cr() {
    validate(&quoted_basic(",", "\r"));
}

#[test]
fn test_quoted_basic_comma_crlf() {
    validate(&quoted_basic(",", "\r\n"));
}

#[test]
fn test_quoted_basic_comma_custom_abc() {
    validate(&quoted_basic(",", "abc"));
}

#[test]
fn test_quoted_basic_comma_custom_aaaaaa() {
    validate(&quoted_basic(",", "aaaaaa"));
}

#[test]
fn test_quoted_basic_space() {
    validate(&quoted_basic(" ", "\n"));
}

#[test]
fn test_quoted_basic_space_cr() {
    validate(&quoted_basic(" ", "\r"));
}

#[test]
fn test_quoted_basic_space_crlf() {
    validate(&quoted_basic(" ", "\r\n"));
}

#[test]
fn test_quoted_basic_space_custom_pokemon() {
    validate(&quoted_basic(" ", "pokemon"));
}

#[test]
fn test_quoted_basic_semicolon() {
    validate(&quoted_basic(";", "\n"));
}

#[test]
fn test_quoted_basic_semicolon_cr() {
    validate(&quoted_basic(";", "\r"));
}

#[test]
fn test_quoted_basic_semicolon_crlf() {
    validate(&quoted_basic(";", "\r\n"));
}

#[test]
fn test_quoted_basic_double_colon() {
    validate(&quoted_basic("::", "\n"));
}

#[test]
fn test_quoted_basic_double_space() {
    validate(&quoted_basic("  ", "\n"));
}

#[test]
fn test_quoted_basic_double_tab() {
    validate(&quoted_basic("\t\t", "\n"));
}

// ---- quoted embedded strings ----

#[test]
fn test_quoted_embedded_strings_comma() {
    validate(&test_quoted_embedded_strings(","));
}

#[test]
fn test_quoted_embedded_strings_space() {
    validate(&test_quoted_embedded_strings(" "));
}

#[test]
fn test_quoted_embedded_strings_tab() {
    validate(&test_quoted_embedded_strings("\t"));
}

#[test]
fn test_quoted_embedded_strings_double_tab() {
    validate(&test_quoted_embedded_strings("\t\t"));
}

#[test]
fn test_quoted_embedded_strings_double_space() {
    validate(&test_quoted_embedded_strings("  "));
}

#[test]
fn test_quoted_embedded_strings_double_colon() {
    validate(&test_quoted_embedded_strings("::"));
}

// ---- quoted type inference ----

#[test]
fn test_quoted_type_inference_comma() {
    validate(&test_quoted_type_inference(",", "\n"));
}

#[test]
fn test_quoted_type_inference_comma_custom_zzz() {
    validate(&test_quoted_type_inference(",", "zzz"));
}

// ---- unicode surrogates ----

#[test]
fn test_unicode_surrogate_pairs() {
    validate(&unicode_surrogate_pairs());
}

// ---- multiline json ----

#[test]
fn test_multiline_json() {
    validate(&multiline_json());
}

// ---- alternate line endings ----

#[test]
fn test_alternate_endline() {
    validate(&alternate_endline_test());
}
