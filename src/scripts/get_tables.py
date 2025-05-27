"""Script for converting json results into LaTeX tables."""

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
    mx = max(vals)
    avg = sum(vals) / len(vals)
    if mx == avg:
        return f"{mx}"
    return f"{mx}({avg:.1f})"

def bold_runtimes(tsqc_times, prr_tsqc_times, bold_both=False, precision=1):
    tsqc_avg = round(sum(tsqc_times) / len(tsqc_times),1)
    prr_tsqc_avg = round(sum(prr_tsqc_times) / len(prr_tsqc_times),1)
    tsqc_fmt = f"{tsqc_avg:.{precision}f}"
    prr_fmt = f"{prr_tsqc_avg:.{precision}f}"
    if bold_both:
        if tsqc_avg < prr_tsqc_avg:
            return f"\\textbf{{{tsqc_fmt}}}", prr_fmt
        elif prr_tsqc_avg < tsqc_avg:
            return tsqc_fmt, f"\\textbf{{{prr_fmt}}}"
        else:
            return f"\\textbf{{{tsqc_fmt}}}", f"\\textbf{{{prr_fmt}}}"
    else:
        return tsqc_fmt, prr_fmt

def bold_max(tsqc_vals, prr_tsqc_vals):
    tsqc_max = max(tsqc_vals)
    prr_tsqc_max = max(prr_tsqc_vals)
    tsqc_obj = format_objective(tsqc_vals)
    prr_tsqc_obj = format_objective(prr_tsqc_vals)
    if tsqc_max > prr_tsqc_max:
        return (f"\\textbf{{{tsqc_obj}}}", f"{prr_tsqc_obj}"), False
    elif tsqc_max < prr_tsqc_max:
        return (f"{tsqc_obj}", f"\\textbf{{{prr_tsqc_obj}}}"), False
    else:
        return (f"\\textbf{{{tsqc_obj}}}", f"\\textbf{{{prr_tsqc_obj}}}"), True

def get_gamma_values(data):
    gamma_values = set()
    for entry in data.values():
        for key in entry:
            try:
                val = float(key)
                gamma_values.add(val)
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

    json_name = os.path.basename(path)            
    base, ext = os.path.splitext(json_name)
    tex_name = base + '.tex' 

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

    # Build table lines
    lines = [header]
    newline = "\\\\"

    for inst in instances:
        info = data[inst]
        nv = info['num_vertices']
        rho = info['density']
        first_gamma = True

        inst_name_formatted = inst.replace("_", " ").lower()
        for gamma in gamma_values:
            tsqc = info[gamma]['tsqc']
            prr_tsqc = info[gamma]['prr_tsqc']

            # Get bolded objectives and whether both are bolded
            (obj_tsqc_str, obj_prr_tsqc_str), bold_rts = bold_max(
                tsqc['objectives'], 
                prr_tsqc['objectives']
            )
            # Get bolded runtimes if needed
            rt_tsqc_str, rt_prr_tsqc_str = bold_runtimes(
                tsqc['runtimes'], prr_tsqc['runtimes'], bold_both=bold_rts
            )
            dash = "--"
            
            if first_gamma:
                prefix = f"{inst_name_formatted} & {nv} & {rho:.3f} & {gamma}"
                first_gamma = False
            else:
                prefix = f"& & & {gamma}"
            
            rest = f"& & {obj_tsqc_str} & {obj_prr_tsqc_str} & {dash} & & {rt_tsqc_str} & {rt_prr_tsqc_str} & {dash}"
            lines.append(f"{prefix} {rest} {newline}")
        lines.append('')

    lines.append(footer)

    # Write to .tex file
    with open(out_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"Wrote filled table to '{out_path}'")