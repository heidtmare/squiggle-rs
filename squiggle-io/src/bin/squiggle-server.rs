//! `squiggle-server [--port n] [--dir path]`: shares projects
//! (`heidtmare.squiggle.sync/1`) and the files behind them with every client.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use squiggle_io::sync::{Hub, serve};

const USAGE: &str = "\
usage: squiggle-server [--port <n>] [--dir <path>]

  --port <n>    TCP port to listen on (default 18100)
  --dir <path>  where shared projects and files are kept (default ./squiggle-server)";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("squiggle-server: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let (mut port, mut dir) = (18100u16, PathBuf::from("squiggle-server"));
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--port" => port = value()?.parse().map_err(|_| "--port: not a port".to_string())?,
            "--dir" => dir = value()?.into(),
            _ => return Err(format!("unknown argument {arg}\n{USAGE}")),
        }
    }
    let hub = Hub::open(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let shared = hub.list();
    let listener = TcpListener::bind(("0.0.0.0", port)).map_err(|e| format!("port {port}: {e}"))?;
    println!("{} shared projects in {} on port {port}", shared.len(), dir.display());
    serve(listener, Arc::new(hub), Arc::new(AtomicBool::new(false))).map_err(|e| e.to_string())
}
