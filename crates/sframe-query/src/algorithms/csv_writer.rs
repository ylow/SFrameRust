//! CSV writer for SFrameRows data.

use sframe_types::error::Result;
use sframe_types::flex_type::FlexType;

use crate::batch::SFrameRows;

/// Quote style for CSV output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteStyle {
    /// Only quote fields that contain the delimiter, quote char, or newlines.
    Minimal,
    /// Quote all fields.
    All,
    /// Quote non-numeric fields.
    NonNumeric,
    /// Never quote (may produce invalid CSV if data contains delimiter).
    None,
}

/// Options for CSV writing.
#[derive(Debug, Clone)]
pub struct CsvWriterOptions {
    pub delimiter: String,
    pub quote_char: char,
    pub escape_char: char,
    pub line_terminator: String,
    pub na_rep: String,
    pub header: bool,
    pub quoting: QuoteStyle,
}

impl Default for CsvWriterOptions {
    fn default() -> Self {
        CsvWriterOptions {
            delimiter: ",".to_string(),
            quote_char: '"',
            escape_char: '\\',
            line_terminator: "\n".to_string(),
            na_rep: String::new(),
            header: true,
            quoting: QuoteStyle::Minimal,
        }
    }
}

/// Write SFrameRows data to CSV string.
pub fn write_csv_string(
    batch: &SFrameRows,
    column_names: &[String],
    options: &CsvWriterOptions,
) -> Result<String> {
    let ncols = batch.num_columns();
    let nrows = batch.num_rows();
    let mut output = String::new();

    // Header
    if options.header {
        for (i, name) in column_names.iter().enumerate() {
            if i > 0 {
                output.push_str(&options.delimiter);
            }
            output.push_str(&quote_field(name, options));
        }
        output.push_str(&options.line_terminator);
    }

    // Data rows
    for row in 0..nrows {
        for col in 0..ncols {
            if col > 0 {
                output.push_str(&options.delimiter);
            }
            let val = batch.column(col).get(row);
            let field = format_value(&val, &options.na_rep);
            output.push_str(&quote_field(&field, options));
        }
        output.push_str(&options.line_terminator);
    }

    Ok(output)
}

/// Write SFrameRows to a CSV file.
pub fn write_csv_file(
    path: &str,
    batch: &SFrameRows,
    column_names: &[String],
    options: &CsvWriterOptions,
) -> Result<()> {
    let content = write_csv_string(batch, column_names, options)?;
    std::fs::write(path, content)
        .map_err(sframe_types::error::SFrameError::Io)
}

fn format_value(val: &FlexType, na_rep: &str) -> String {
    match val {
        FlexType::Undefined => na_rep.to_string(),
        FlexType::Integer(i) => i.to_string(),
        FlexType::Float(f) => {
            if f.is_nan() {
                "nan".to_string()
            } else if f.is_infinite() {
                if *f > 0.0 { "inf".to_string() } else { "-inf".to_string() }
            } else {
                format!("{}", f)
            }
        }
        FlexType::String(s) => s.to_string(),
        FlexType::Vector(v) => {
            let elements: Vec<String> = v.iter().map(|f| format!("{}", f)).collect();
            format!("[{}]", elements.join(", "))
        }
        FlexType::List(l) => {
            let elements: Vec<String> = l.iter().map(|v| format_value(v, na_rep)).collect();
            format!("[{}]", elements.join(", "))
        }
        FlexType::Dict(d) => {
            let entries: Vec<String> = d
                .iter()
                .map(|(k, v)| format!("{}: {}", format_value(k, na_rep), format_value(v, na_rep)))
                .collect();
            format!("{{{}}}", entries.join(", "))
        }
        FlexType::DateTime(dt) => format!("{}", dt.posix_timestamp),
    }
}

fn quote_field(field: &str, options: &CsvWriterOptions) -> String {
    let needs_quoting = match options.quoting {
        QuoteStyle::All => true,
        QuoteStyle::None => false,
        QuoteStyle::NonNumeric => {
            // Check if the field is purely numeric
            field.parse::<f64>().is_err()
        }
        QuoteStyle::Minimal => {
            field.contains(&*options.delimiter)
                || field.contains(options.quote_char)
                || field.contains('\n')
                || field.contains('\r')
        }
    };

    if needs_quoting {
        let escaped = field.replace(
            options.quote_char,
            &format!("{}{}", options.quote_char, options.quote_char),
        );
        format!("{}{}{}", options.quote_char, escaped, options.quote_char)
    } else {
        field.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::ColumnData;
    use sframe_types::flex_type::FlexTypeEnum;

    #[test]
    fn test_basic_csv_write() {
        let batch = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(1), Some(2), Some(3)]),
            ColumnData::String(vec![Some("alice".into()), Some("bob".into()), Some("charlie".into())]),
        ]).unwrap();

        let names = vec!["id".to_string(), "name".to_string()];
        let csv = write_csv_string(&batch, &names, &CsvWriterOptions::default()).unwrap();

        assert!(csv.starts_with("id,name\n"));
        assert!(csv.contains("1,alice\n"));
        assert!(csv.contains("2,bob\n"));
        assert!(csv.contains("3,charlie\n"));
    }

    #[test]
    fn test_csv_quoting() {
        let batch = SFrameRows::new(vec![
            ColumnData::String(vec![Some("hello, world".into()), Some("no comma".into())]),
        ]).unwrap();

        let names = vec!["text".to_string()];
        let csv = write_csv_string(&batch, &names, &CsvWriterOptions::default()).unwrap();

        assert!(csv.contains("\"hello, world\""));
        assert!(csv.contains("no comma"));
    }

    #[test]
    fn test_csv_na_values() {
        let batch = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(1), None, Some(3)]),
        ]).unwrap();

        let names = vec!["x".to_string()];
        let mut opts = CsvWriterOptions::default();
        opts.na_rep = "NA".to_string();
        let csv = write_csv_string(&batch, &names, &opts).unwrap();

        assert!(csv.contains("NA"));
    }

    #[test]
    fn test_csv_no_header() {
        let batch = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(1)]),
        ]).unwrap();

        let names = vec!["x".to_string()];
        let mut opts = CsvWriterOptions::default();
        opts.header = false;
        let csv = write_csv_string(&batch, &names, &opts).unwrap();

        assert_eq!(csv, "1\n");
    }
}
