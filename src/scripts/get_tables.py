"""Script for converting json results into LaTeX tables, now including DeepTSQC."""

import json, sys, os, yaml

CONFIG_PATHS = {
    "results/dimacs/dimacs.json": "src/config/dimacs.yml",
    "results/real-life/real_life.json": "src/config/real-life.yml"
}

def load_config(config_path):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from '{config_path}'")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file '{config_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        sys.exit(1)

def format_objective(vals):
    """Show max if equal to avg, else max(avg)."""
    mx = max(vals)
    avg = sum(vals) / len(vals)
    if mx == avg:
        return f"{mx}"
    return f"{mx}({avg:.1f})"

def bold_max(tsqc_vals, prr_vals, deep_vals):
    """
    Bold any method whose max objective == global max.
    Return triple of formatted objective strings and a flag if ≥2 tie.
    """
    # raw maxima
    ts_max, pr_max, dp_max = max(tsqc_vals), max(prr_vals), max(deep_vals)
    global_max = max(ts_max, pr_max, dp_max)

    # formatted objective text
    ts_obj = format_objective(tsqc_vals)
    pr_obj = format_objective(prr_vals)
    dp_obj = format_objective(deep_vals)

    # decide bolding
    ts_best = ts_max == global_max
    pr_best = pr_max == global_max
    dp_best = dp_max == global_max

    ts_str = f"\\textbf{{{ts_obj}}}" if ts_best else ts_obj
    pr_str = f"\\textbf{{{pr_obj}}}" if pr_best else pr_obj
    dp_str = f"\\textbf{{{dp_obj}}}" if dp_best else dp_obj

    ts_str = "TL" if ts_obj == "0" else ts_str
    pr_str = "TL" if ts_obj == "0" else pr_str
    dp_str = "TL" if ts_obj == "0" else dp_str

    # if two or more share top objective, we will bold the fastest runtime
    bold_rts = sum((ts_best, pr_best, dp_best)) > 1

    return (ts_str, pr_str, dp_str), bold_rts

def bold_runtimes(tsqc_times, prr_times, deep_times, bold_flag=False, precision=1):
    """
    Compute average runtimes, format to given precision,
    and if bold_flag, bold those equal to the minimum.
    """
    ts_avg = round(sum(tsqc_times) / len(tsqc_times), precision)
    pr_avg = round(sum(prr_times) / len(prr_times), precision)
    dp_avg = round(sum(deep_times) / len(deep_times), precision)

    ts_fmt = f"{ts_avg:.{precision}f}"
    pr_fmt = f"{pr_avg:.{precision}f}"
    dp_fmt = f"{dp_avg:.{precision}f}"

    if bold_flag:
        mn = min(ts_avg, pr_avg, dp_avg)
        if ts_avg == mn: ts_fmt = f"\\textbf{{{ts_fmt}}}"
        if pr_avg == mn: pr_fmt = f"\\textbf{{{pr_fmt}}}"
        if dp_avg == mn: dp_fmt = f"\\textbf{{{dp_fmt}}}"

    return ts_fmt, pr_fmt, dp_fmt

def get_gamma_values(data):
    gamma_values = set()
    for entry in data.values():
        for key in entry:
            try:
                gamma_values.add(float(key))
            except ValueError:
                pass
    return sorted(gamma_values)

for path in ("results/dimacs/dimacs.json", "results/real-life/real_life.json"):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"No file found at {path!r}.")
        continue

    instances = sorted(data.keys())
    config = load_config(CONFIG_PATHS[path])
    gamma_values = [str(gam) for gam in config["gamma"]]

    out_dir = os.path.dirname(path)
    tex_name = os.path.splitext(os.path.basename(path))[0] + '.tex'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, tex_name)

    header = r"""\begin{table}[ht]
    \caption{Numerical results for all DIMACS instances.}
    \label{tab:results}
    \centering
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{l c c c c c c c c c c c}
    \toprule
    \toprule
    \multicolumn{4}{c}{instance} & & \multicolumn{3}{c}{$\omega_{\gamma}$: max(avg)} & &
    \multicolumn{3}{c}{$t$: avg} \\
    \cmidrule{1-4} \cmidrule{6-8} \cmidrule{10-12}
    name & $|V|$ & $\rho$ & $\gamma$ & & \texttt{TSQC} & \texttt{PRR-TSQC} & \texttt{DeepTSQC} & & \texttt{TSQC} & \texttt{PRR-TSQC} & \texttt{DeepTSQC} \\
    \midrule"""

    footer = r"""
    \bottomrule
    \bottomrule
    \addlinespace[1pt]
    \multicolumn{12}{l}{The maximum quasi-clique size found is denoted by $\omega_{\gamma}$, and $t$ denotes the runtime in seconds.} \\
    \end{tabular}
    }
    \end{table}"""

    lines = [header]
    newline = r"\\"

    for i, inst in enumerate(instances):
        info = data[inst]
        nv = info['num_vertices']
        rho = info['density']
        first_gamma = True
        inst_name = inst.replace("_", " ").lower()

        for gamma in gamma_values:
            tsqc = info[gamma]['tsqc']
            prr  = info[gamma]['prr_tsqc']
            dp   = info[gamma]['deeptsqc']

            # determine bolding for objectives
            (obj_ts, obj_pr, obj_dp), tie_on_obj = bold_max(
                tsqc['objectives'],
                prr['objectives'],
                dp['objectives']
            )
            # determine bolding for runtimes
            rt_ts, rt_pr, rt_dp = bold_runtimes(
                tsqc['runtimes'],
                prr['runtimes'],
                dp['runtimes'],
                bold_flag=tie_on_obj
            )

            if first_gamma:
                prefix = f"{inst_name} & {nv} & {rho:.3f} & {gamma}"
                first_gamma = False
            else:
                prefix = "& & & " + gamma
            rest = (
                f"& & {obj_ts} & {obj_pr} & {obj_dp} "
                f"& & {rt_ts} & {rt_pr} & {rt_dp}"
            )
            lines.append(f"{prefix} {rest} {newline}")

        if i < len(instances) - 1:lines.append(r"\midrule")

    lines.append(footer)

    with open(out_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"Wrote filled table to '{out_path}'")
