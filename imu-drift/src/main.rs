// SPDX-License-Identifier: AGPL-3.0-or-later

use imu_drift::allan::{allan_variance, fit_arw_bias_instab_rrw};
use std::{env, fs, process};

#[derive(Debug)]
struct AllanArgs {
    samples_path: String,
    dt_s: Option<f64>,
    rate_hz: Option<f64>,
    column: Option<String>,
    taus: Option<Vec<f64>>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("allan-variance") => run_allan_variance(parse_allan_args(args.collect())?),
        Some("-h") | Some("--help") | None => {
            print_usage();
            Ok(())
        }
        Some(cmd) => Err(format!("unknown command: {cmd}\n\n{}", usage())),
    }
}

fn parse_allan_args(raw: Vec<String>) -> Result<AllanArgs, String> {
    let mut out = AllanArgs {
        samples_path: String::new(),
        dt_s: None,
        rate_hz: None,
        column: None,
        taus: None,
    };
    let mut i = 0usize;
    while i < raw.len() {
        match raw[i].as_str() {
            "--samples" => {
                i += 1;
                out.samples_path = raw.get(i).ok_or("--samples requires a path")?.clone();
            }
            "--dt" => {
                i += 1;
                out.dt_s = Some(parse_positive(raw.get(i), "--dt")?);
            }
            "--rate-hz" => {
                i += 1;
                out.rate_hz = Some(parse_positive(raw.get(i), "--rate-hz")?);
            }
            "--column" => {
                i += 1;
                out.column = Some(
                    raw.get(i)
                        .ok_or("--column requires a name or zero-based index")?
                        .clone(),
                );
            }
            "--taus" => {
                i += 1;
                let value = raw.get(i).ok_or("--taus requires a comma-separated list")?;
                out.taus = Some(
                    value
                        .split(',')
                        .map(|part| {
                            part.trim()
                                .parse::<f64>()
                                .map_err(|_| format!("invalid tau: {part}"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            other => return Err(format!("unknown option: {other}")),
        }
        i += 1;
    }
    if out.samples_path.is_empty() {
        return Err("--samples is required".to_string());
    }
    if out.dt_s.is_some() && out.rate_hz.is_some() {
        return Err("use either --dt or --rate-hz, not both".to_string());
    }
    Ok(out)
}

fn parse_positive(value: Option<&String>, name: &str) -> Result<f64, String> {
    let parsed = value
        .ok_or_else(|| format!("{name} requires a value"))?
        .parse::<f64>()
        .map_err(|_| format!("{name} must be numeric"))?;
    if parsed <= 0.0 || !parsed.is_finite() {
        return Err(format!("{name} must be positive and finite"));
    }
    Ok(parsed)
}

fn run_allan_variance(args: AllanArgs) -> Result<(), String> {
    let text = fs::read_to_string(&args.samples_path)
        .map_err(|err| format!("failed to read {}: {err}", args.samples_path))?;
    let table = parse_table(&text)?;
    let column_index = select_column(&table, args.column.as_deref())?;
    let samples = table
        .rows
        .iter()
        .filter_map(|row| row.get(column_index).copied())
        .collect::<Vec<_>>();
    if samples.len() < 4 {
        return Err("need at least 4 numeric samples".to_string());
    }

    let dt_s = if let Some(dt) = args.dt_s {
        dt
    } else if let Some(rate_hz) = args.rate_hz {
        1.0 / rate_hz
    } else {
        infer_dt_s(&table, column_index).unwrap_or(1.0)
    };

    let taus = args.taus.unwrap_or_else(|| default_taus(dt_s, samples.len()));
    if taus.is_empty() {
        return Err("no usable taus for this dataset".to_string());
    }
    let mut sigma_a_sq = vec![0.0_f64; taus.len()];
    allan_variance(&samples, dt_s, &taus, &mut sigma_a_sq)
        .map_err(|err| format!("allan variance failed: {err:?}"))?;
    let fit = fit_arw_bias_instab_rrw(&taus, &sigma_a_sq)
        .map_err(|err| format!("noise fit failed: {err:?}"))?;

    println!("{{");
    println!("  \"samples\": {},", samples.len());
    println!("  \"dt_s\": {},", dt_s);
    println!("  \"column\": \"{}\",", table.column_name(column_index));
    println!("  \"allan\": [");
    for (i, (&tau, &variance)) in taus.iter().zip(sigma_a_sq.iter()).enumerate() {
        let comma = if i + 1 == taus.len() { "" } else { "," };
        println!(
            "    {{\"tau_s\": {}, \"variance\": {}, \"deviation\": {}}}{}",
            tau,
            variance,
            variance.max(0.0).sqrt(),
            comma
        );
    }
    println!("  ],");
    println!("  \"fit\": {{");
    println!("    \"arw\": {},", fit.arw);
    println!("    \"bias_instab\": {},", fit.bias_instab);
    println!("    \"rrw\": {}", fit.rrw);
    println!("  }}");
    println!("}}");
    Ok(())
}

struct Table {
    headers: Option<Vec<String>>,
    rows: Vec<Vec<f64>>,
}

impl Table {
    fn column_name(&self, index: usize) -> String {
        self.headers
            .as_ref()
            .and_then(|headers| headers.get(index))
            .cloned()
            .unwrap_or_else(|| index.to_string())
    }
}

fn parse_table(text: &str) -> Result<Table, String> {
    let mut headers = None;
    let mut rows = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields = split_fields(trimmed);
        if fields.is_empty() {
            continue;
        }
        let parsed = fields
            .iter()
            .map(|field| field.parse::<f64>())
            .collect::<Result<Vec<_>, _>>();
        match parsed {
            Ok(row) => rows.push(row),
            Err(_) if rows.is_empty() && headers.is_none() => {
                headers = Some(fields.into_iter().map(str::to_string).collect());
            }
            Err(_) => return Err(format!("non-numeric data row: {trimmed}")),
        }
    }
    if rows.is_empty() {
        return Err("sample file contains no numeric rows".to_string());
    }
    Ok(Table { headers, rows })
}

fn split_fields(line: &str) -> Vec<&str> {
    line.split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .collect()
}

fn select_column(table: &Table, requested: Option<&str>) -> Result<usize, String> {
    if let Some(value) = requested {
        if let Ok(index) = value.parse::<usize>() {
            return validate_column(table, index);
        }
        if let Some(headers) = &table.headers {
            if let Some(index) = headers.iter().position(|header| header == value) {
                return validate_column(table, index);
            }
        }
        return Err(format!("unknown column: {value}"));
    }

    if let Some(headers) = &table.headers {
        for (i, header) in headers.iter().enumerate() {
            let normalized = header.to_ascii_lowercase();
            if !matches!(normalized.as_str(), "time" | "t" | "timestamp" | "timestamp_s") {
                return validate_column(table, i);
            }
        }
    }
    validate_column(table, 0)
}

fn validate_column(table: &Table, index: usize) -> Result<usize, String> {
    if table.rows.iter().all(|row| index < row.len()) {
        Ok(index)
    } else {
        Err(format!("column index out of range: {index}"))
    }
}

fn infer_dt_s(table: &Table, sample_column: usize) -> Option<f64> {
    let headers = table.headers.as_ref()?;
    let time_index = headers.iter().position(|header| {
        matches!(
            header.to_ascii_lowercase().as_str(),
            "time" | "t" | "timestamp" | "timestamp_s"
        )
    })?;
    if time_index == sample_column || table.rows.len() < 2 {
        return None;
    }
    let mut diffs = table
        .rows
        .windows(2)
        .filter_map(|pair| Some(pair[1].get(time_index)? - pair[0].get(time_index)?))
        .filter(|dt| *dt > 0.0 && dt.is_finite())
        .collect::<Vec<_>>();
    if diffs.is_empty() {
        return None;
    }
    diffs.sort_by(|a, b| a.total_cmp(b));
    Some(diffs[diffs.len() / 2])
}

fn default_taus(dt_s: f64, samples: usize) -> Vec<f64> {
    let max_tau = dt_s * samples as f64 / 2.0;
    let multipliers = [1.0, 2.0, 5.0];
    let mut taus = Vec::new();
    let mut decade = 1.0;
    while dt_s * decade <= max_tau {
        for multiplier in multipliers {
            let tau = dt_s * decade * multiplier;
            if tau <= max_tau {
                taus.push(tau);
            }
        }
        decade *= 10.0;
    }
    taus.sort_by(|a, b| a.total_cmp(b));
    taus.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON);
    taus
}

fn print_usage() {
    println!("{}", usage());
}

fn usage() -> &'static str {
    "Usage:
  imu-drift allan-variance --samples <csv-or-whitespace-file> [--dt <seconds> | --rate-hz <hz>] [--column <name-or-index>] [--taus <t1,t2,...>]

Examples:
  cargo run --release -- allan-variance --samples imu.csv --dt 0.01 --column gyro_z
  cargo run --release -- allan-variance --samples gyro_z.txt --rate-hz 100"
}
