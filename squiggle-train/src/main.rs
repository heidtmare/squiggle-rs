//! `squiggle-train <dataset> --out <package> [options]`: see `USAGE`.

use std::path::PathBuf;
use std::process::ExitCode;

use squiggle_models::Kind;
use squiggle_train::{Dataset, Level, Options, predictions, train};

const USAGE: &str = "\
usage: squiggle-train <dataset-dir> --out <package-dir> [options]

  --model <kind>        logistic | svm | knn | tree | forest (default forest)
  --name <name>         model name (default: the kind)
  --version <n>         model version (default 1)
  --level <level>       leaf (analyst label) | top (root ancestor) (default leaf)
  --no-noise            leave out rejected events instead of training a noise class
  --features <a,b,…>    event features to use (default all)
  --seed <n>            random seed (default 1)
  --folds <n>           cross-validation folds, at most the smallest class (default 5)
  --l2 <x> --iterations <n> --rate <x> --k <n> --max-depth <n> --min-leaf <n> --trees <n>
                        hyperparameters
  --predictions <file>  also write each event's probabilities as CSV";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("squiggle-train: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let mut options = Options::default();
    let mut dataset: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut csv: Option<PathBuf> = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        fn number<T: std::str::FromStr>(arg: &str, v: String) -> Result<T, String> {
            v.parse().map_err(|_| format!("{arg}: not a number: {v}"))
        }
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--out" => out = Some(value()?.into()),
            "--model" => {
                let v = value()?;
                options.kind = Kind::parse(&v).ok_or_else(|| format!("unknown model {v}"))?;
            }
            "--name" => options.name = Some(value()?),
            "--version" => options.version = number(&arg, value()?)?,
            "--level" => {
                options.level = match value()?.as_str() {
                    "leaf" => Level::Leaf,
                    "top" => Level::Top,
                    v => return Err(format!("unknown level {v}")),
                }
            }
            "--no-noise" => options.noise = false,
            "--features" => options.features = Some(value()?.split(',').map(|s| s.trim().to_string()).collect()),
            "--seed" => options.seed = number(&arg, value()?)?,
            "--folds" => options.folds = number(&arg, value()?)?,
            "--l2" => options.params.l2 = number(&arg, value()?)?,
            "--iterations" => options.params.iterations = number(&arg, value()?)?,
            "--rate" => options.params.rate = number(&arg, value()?)?,
            "--k" => options.params.k = number(&arg, value()?)?,
            "--max-depth" => options.params.max_depth = number(&arg, value()?)?,
            "--min-leaf" => options.params.min_leaf = number(&arg, value()?)?,
            "--trees" => options.params.trees = number(&arg, value()?)?,
            "--predictions" => csv = Some(value()?.into()),
            _ if arg.starts_with('-') => return Err(format!("unknown option {arg}\n{USAGE}")),
            _ if dataset.is_none() => dataset = Some(arg.into()),
            _ => return Err(format!("unexpected argument {arg}\n{USAGE}")),
        }
    }
    let (Some(dataset), Some(out)) = (dataset, out) else {
        return Err(USAGE.to_string());
    };

    let dataset = Dataset::load(&dataset)?;
    let package = train(&dataset, &options)?;
    package.save(&out)?;

    let spec = &package.spec;
    let t = &spec.training;
    println!("{} → {}", spec.id, out.display());
    let classes: Vec<String> = t.classes.iter().map(|c| format!("{} {}", c.events, c.label)).collect();
    println!("trained on {} events: {}", t.events, classes.join(", "));
    println!("fit accuracy {:.3}, log loss {:.3}", t.fit.accuracy, t.fit.log_loss);
    if t.folds > 0 {
        let cv = &t.cross_validation;
        println!("{}-fold accuracy {:.3}, log loss {:.3}; confusion (rows true, columns predicted):", t.folds, cv.accuracy, cv.log_loss);
        let width = spec.output.labels.iter().map(String::len).max().unwrap_or(0);
        for (label, row) in spec.output.labels.iter().zip(&cv.confusion) {
            let cells: Vec<String> = row.iter().map(|n| format!("{n:>5}")).collect();
            println!("  {label:>width$} {}", cells.join(""));
        }
    } else {
        println!("too few events per class to cross-validate");
    }

    if let Some(path) = csv {
        let mut text = format!("event,{}\n", spec.output.labels.join(","));
        for (event, p) in predictions(&package, &dataset) {
            let cells: Vec<String> = p.iter().map(|v| v.to_string()).collect();
            text.push_str(&format!("{event},{}\n", cells.join(",")));
        }
        std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
        println!("predictions → {}", path.display());
    }
    Ok(())
}
