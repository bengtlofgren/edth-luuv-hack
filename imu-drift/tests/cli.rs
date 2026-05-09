// SPDX-License-Identifier: AGPL-3.0-or-later

use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn allan_variance_cli_reads_csv_and_prints_fit() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("imu-drift-cli-{unique}.csv"));
    let mut csv = String::from("time,gyro_z\n");
    for i in 0..200 {
        let t = i as f64 * 0.01;
        let sample = (i as f64 * 0.1).sin() * 0.01;
        csv.push_str(&format!("{t},{sample}\n"));
    }
    fs::write(&path, csv).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_imu-drift"))
        .args([
            "allan-variance",
            "--samples",
            path.to_str().unwrap(),
            "--column",
            "gyro_z",
            "--taus",
            "0.01,0.02,0.05,0.1",
        ])
        .output()
        .unwrap();

    let _ = fs::remove_file(path);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"allan\""));
    assert!(stdout.contains("\"fit\""));
    assert!(stdout.contains("\"arw\""));
}
