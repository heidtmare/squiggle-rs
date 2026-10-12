//! `squiggle-stream <files…> [--port n] [--speed x] [--chunk s]`: replays
//! waveform files as a live `heidtmare.squiggle.stream/1` feed, paced by
//! their timestamps, to every client that connects.

use std::net::TcpListener;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use squiggle_io::stream::{ReplayOptions, serve};

const USAGE: &str = "\
usage: squiggle-stream <file>… [--port <n>] [--speed <x>] [--chunk <s>]

  --port <n>    TCP port to listen on (default 18000)
  --speed <x>   stream seconds per second; 0 sends as fast as possible (default 1)
  --chunk <s>   seconds per chunk (default 0.5)";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("squiggle-stream: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let (mut files, mut port, mut options) = (Vec::new(), 18000u16, ReplayOptions::default());
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        let number = |v: String| v.parse::<f64>().map_err(|_| format!("not a number: {v}"));
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--port" => port = value()?.parse().map_err(|_| "--port: not a port".to_string())?,
            "--speed" => {
                let speed = number(value()?)?;
                options.speed = if speed > 0.0 { speed } else { f64::INFINITY };
            }
            "--chunk" => options.chunk_s = number(value()?)?.max(1e-6),
            _ if arg.starts_with('-') => return Err(format!("unknown option {arg}\n{USAGE}")),
            _ => files.push(arg),
        }
    }
    if files.is_empty() {
        return Err(USAGE.to_string());
    }
    let mut recordings = Vec::new();
    for file in &files {
        let read = squiggle_io::read_file(std::path::Path::new(file)).map_err(|e| format!("{file}: {e}"))?;
        recordings.extend(read.recordings);
    }
    // Channel numbers must be unique across files.
    for (i, r) in recordings.iter_mut().enumerate() {
        r.channel = i as u32;
    }
    let listener = TcpListener::bind(("0.0.0.0", port)).map_err(|e| format!("port {port}: {e}"))?;
    println!("{} channels on port {port} at ×{}; each client gets the whole replay", recordings.len(), options.speed);
    serve(listener, Arc::new(recordings), options, Arc::new(AtomicBool::new(false))).map_err(|e| e.to_string())
}
