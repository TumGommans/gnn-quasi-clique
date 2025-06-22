"""
Script for converting json results into the LaTeX tables included in the paper:
1) The numerical results for the DIMACS and real-life graphs.
2) The table describing hyperparamerers in Appendix B.
"""

import json, sys, os, yaml
from typing import Union

# ----------------------------------------------------------------------------------------------------
# CODE FOR NUMERIC RESULTS TABLES FOR DIMACS AND REAL-LIFE GRAPHS
# ----------------------------------------------------------------------------------------------------

CONFIG_PATHS = {
    "results/dimacs/dimacs.json": "src/config/run/dimacs.yml",
    "results/real-life/real_life.json": "src/config/run/real-life.yml"
}

def load_config(config_path: str):
    """Loads configuration from a YAML file.
    
    args:
        config_path: the path at which the config is located."""
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

def format_objective(vals: list[int]):
    """Show max if equal to avg, else max(avg).
    
    args:
        vals: list of objective values"""
    mx = max(vals)
    avg = sum(vals) / len(vals)
    if mx == avg:
        return f"{mx}"
    return f"{mx}({avg:.1f})"

def bold_max(
    tsqc_vals: list[int], 
    deep_vals: list[int]
) -> tuple[tuple[str], bool, tuple[bool]]:
    """
    Bold methods according to:
    - If one max is higher, bold that one.
    - If tied on max and one average is higher, bold that one.
    - If tied on both max and avg, bold both.
    Also, if the average objective for a method is 0, set its objective to "TL"
    and mark its runtime as "TL".

    args:
        tsqc_vals: objective values for TSQC
        deep_vals: objective values for DeepTSQC

    returns:
      - (ts_str, dp_str): objective strings (with \\textbf{} if bolded, or "TL").
      - tie_runtimes: bool indicating whether to bold runtimes (only when max and avg are exactly tied).
      - (ts_tl, dp_tl): bools indicating if runtime should be "TL" for TSQC / DeepTSQC.
    """
    ts_max = max(tsqc_vals)
    dp_max = max(deep_vals)
    ts_avg = sum(tsqc_vals) / len(tsqc_vals)
    dp_avg = sum(deep_vals) / len(deep_vals)

    # Determine objective strings (without bold) or "TL" if avg == 0
    if ts_avg == 0:
        ts_str = "TL"
    else:
        ts_str = format_objective(tsqc_vals)

    if dp_avg == 0:
        dp_str = "TL"
    else:
        dp_str = format_objective(deep_vals)

    # Determine bold flags for objectives
    ts_bold = False
    dp_bold = False

    if ts_str == "TL" and dp_str != "TL":
        # TSQC timed out, only DeepTSQC can be bold if its max > 0
        if dp_max > ts_max:
            dp_bold = True
    elif dp_str == "TL" and ts_str != "TL":
        # DeepTSQC timed out, only TSQC can be bold if its max > 0
        if ts_max > dp_max:
            ts_bold = True
    else:
        # Neither or both are not "TL"
        if ts_max > dp_max:
            ts_bold = True
        elif dp_max > ts_max:
            dp_bold = True
        else:
            # ts_max == dp_max
            if ts_avg > dp_avg:
                ts_bold = True
            elif dp_avg > ts_avg:
                dp_bold = True
            else:
                # ts_avg == dp_avg
                ts_bold = True
                dp_bold = True

    # Apply bold formatting (but skip if string is "TL")
    if ts_bold and ts_str != "TL":
        ts_str = f"\\textbf{{{ts_str}}}"
    if dp_bold and dp_str != "TL":
        dp_str = f"\\textbf{{{dp_str}}}"

    # Only when max and avg are exactly tied (and neither timed out).
    tie_runtimes = (ts_max == dp_max) and (ts_avg == dp_avg) and (ts_str != "TL") and (dp_str != "TL")

    # Determine timeout flags for runtimes
    ts_tl = (ts_avg == 0)
    dp_tl = (dp_avg == 0)

    return (ts_str, dp_str), tie_runtimes, (ts_tl, dp_tl)

def bold_runtimes(
    tsqc_times: list[float], 
    deep_times: list[float], 
    bold_flag: bool = False, 
    precision: int = 1
) -> tuple[str]:
    """
    Compute average runtimes, format to given precision,
    and if bold_flag, bold those equal to the minimum.

    args:
        tsqc_times: runtimes for TSQC
        deep_times: runtimes for DeepTSQC
        bold_flag: bool indicating whether objective metrics were equal
    
    returns:
        ts_fmt: formatted average runtime for TSQC
        dp_fmt: formatted average runtime for DeepTSQC
    """
    ts_avg = round(sum(tsqc_times) / len(tsqc_times), precision)
    dp_avg = round(sum(deep_times) / len(deep_times), precision)

    ts_fmt = f"{ts_avg:.{precision}f}"
    dp_fmt = f"{dp_avg:.{precision}f}"

    if bold_flag:
        mn = min(ts_avg, dp_avg)
        if ts_avg == mn:
            ts_fmt = f"\\textbf{{{ts_fmt}}}"
        if dp_avg == mn:
            dp_fmt = f"\\textbf{{{dp_fmt}}}"

    return ts_fmt, dp_fmt

def get_gamma_values(data: dict) -> list[float]:
    """Extract the gamma values used for this data dictionary.
    
    args:
        data: dictionary containing all results for either DIMACS or real-life
    """
    gamma_values = set()
    for entry in data.values():
        for key in entry:
            try:
                gamma_values.add(float(key))
            except ValueError:
                pass
    return sorted(gamma_values)

def classify_instance(info: dict, gamma_values: list[float]) -> bool:
    """
    Decide whether an instance belongs in the regular table or the appendix.

    Rule logic:
      1) If for any gamma, one method has a strictly better max objective
         OR strictly better average objective, return True (regular).
      2) Otherwise (objectives identical for all gamma):
         - Compute per-gamma average runtimes, but exclude any gamma where "TL".
         - Let diffs = [|ts_avg_rt - dp_avg_rt| for those gammas].
         - If len(diffs)>0 and (sum(diffs)/len(diffs) > 0.2), return True (regular).
      3) Else, return False (appendix).
    
    args:
        info: results dictionary for a given instance
        gamma_values: the values of gamma present in the info dict
    
    returns:
        bool: whether the instance belongs to the regular table
    """
    # Check objective differences
    for gamma in gamma_values:
        tsqc = info[gamma]['tsqc']
        dp = info[gamma]['deeptsqc']
        ts_max = max(tsqc['objectives'])
        dp_max = max(dp['objectives'])
        ts_avg_obj = sum(tsqc['objectives']) / len(tsqc['objectives'])
        dp_avg_obj = sum(dp['objectives']) / len(dp['objectives'])
        if ts_max != dp_max or ts_avg_obj != dp_avg_obj:
            return True
    
    # Now check runtimes
    diffs = []
    for gamma in gamma_values:
        tsqc = info[gamma]['tsqc']
        dp = info[gamma]['deeptsqc']
        ts_avg_obj = sum(tsqc['objectives']) / len(tsqc['objectives'])
        dp_avg_obj = sum(dp['objectives']) / len(dp['objectives'])
        if ts_avg_obj == 0 or dp_avg_obj == 0:
            continue
        ts_avg_rt = sum(tsqc['runtimes']) / len(tsqc['runtimes'])
        dp_avg_rt = sum(dp['runtimes']) / len(dp['runtimes'])
        diffs.append(abs(ts_avg_rt - dp_avg_rt))
    if len(diffs) > 0:
        avg_diff = sum(diffs) / len(diffs)
        if avg_diff > 0.2:
            return True
    
    # Appendix instance
    return False

def write_table(
    data: dict, 
    instances: list[str], 
    gamma_values: list[float], 
    out_path: str, 
    caption_text: str, 
    label_text: str
):
    """
    Write a .tex file for the given instances list with a custom caption.

    args:
        data: the full results dictionary for either the DIMACS or real-life instances
        instances: the list of instance names belonging to the same table (either regular/appendix)
        gamma_values: list of gamma values for which this class of graphs is run
        out_path: path to write the .tex file to
        caption_text: caption for in the LaTeX code
        label_text: label for in the LaTeX code
    """
    header = rf"""\begin{{table}}[h!]
    \caption{{{caption_text}}}
    \label{{{label_text}}}
    \centering
    \begin{{tabular}}{{l c c c c c c c c c}}
    \toprule
    \toprule
    \multicolumn{{4}}{{c}}{{instance}} & & \multicolumn{{2}}{{c}}{{$\omega_{{\gamma}}$: max(avg)}} & &
    \multicolumn{{2}}{{c}}{{$t$: avg}} \\
    \cmidrule{{1-4}} \cmidrule{{6-7}} \cmidrule{{9-10}}
    name & $|V|$ & $\rho$ & $\gamma$ & & \texttt{{TSQC}} & \texttt{{DeepTSQC}} & & \texttt{{TSQC}} & \texttt{{DeepTSQC}} \\
    \midrule"""

    footer = r"""
    \bottomrule
    \bottomrule
    \end{tabular}
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
            dp = info[gamma]['deeptsqc']

            # determine bolding for objectives and TL flags
            (obj_ts, obj_dp), tie_on_obj, (ts_obj_tl, dp_obj_tl) = bold_max(
                tsqc['objectives'],
                dp['objectives']
            )
            # determine bolding for runtimes
            rt_ts, rt_dp = bold_runtimes(
                tsqc['runtimes'],
                dp['runtimes'],
                bold_flag=tie_on_obj
            )
            # if objective avg was 0 (TL), set runtime to "TL" as well
            if ts_obj_tl:
                rt_ts = "TL"
            if dp_obj_tl:
                rt_dp = "TL"

            if first_gamma:
                prefix = f"{inst_name} & {nv} & {rho:.3f} & {gamma}"
                first_gamma = False
            else:
                prefix = "& & & " + gamma
            rest = (
                f"& & {obj_ts} & {obj_dp} "
                f"& & {rt_ts} & {rt_dp}"
            )
            lines.append(f"{prefix} {rest} {newline}")

        if i < len(instances) - 1:
            lines.append(r"\midrule")

    lines.append(footer)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Wrote table to '{out_path}'")

# for path in ("results/dimacs/dimacs.json", "results/real-life/real_life.json"):
#     try:
#         with open(path, 'r') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"No file found at {path!r}.")
#         continue

#     instances = sorted(data.keys())
#     config = load_config(CONFIG_PATHS[path])
#     gamma_values = [str(gam) for gam in config["gamma"]]

#     # Partition instances into regular vs appendix
#     regular_instances = []
#     appendix_instances = []
#     for inst in instances:
#         info = data[inst]
#         if classify_instance(info, gamma_values):
#             regular_instances.append(inst)
#         else:
#             appendix_instances.append(inst)

#     base_dir = os.path.dirname(path)
#     base_name = os.path.splitext(os.path.basename(path))[0]

#     if base_name == "dimacs":
#         regular_caption = "Numerical results for DIMACS instances."
#         appendix_caption = "Numerical results for DIMACS instances continued."
#         regular_label   = "tab: dimacs"
#         appendix_label  = "tab: dimacs cont"
#     elif base_name == "real_life":
#         regular_caption = "Numerical results for real life instances."
#         appendix_caption = "Numerical results for real life instances continued."
#         regular_label   = "tab: real life"
#         appendix_label  = "tab: real life cont"
#     else:
#         human = base_name.replace("_", " ")
#         regular_caption = f"Numerical results for {human} instances."
#         appendix_caption = f"Numerical results for {human} instances continued."
#         regular_label   = f"tab: {human}"
#         appendix_label  = f"tab: {human} cont"

#     # Write regular table
#     regular_tex = os.path.join(base_dir, f"{base_name}.tex")
#     write_table(data, regular_instances, gamma_values,
#                 regular_tex, regular_caption, regular_label)

#     # Write appendix table
#     appendix_tex = os.path.join(base_dir, f"{base_name}_appendix.tex")
#     write_table(data, appendix_instances, gamma_values,
#                 appendix_tex, appendix_caption, appendix_label)

# ----------------------------------------------------------------------------------------------------
# CODE FOR HYPERPARAMERER TABLE
# ----------------------------------------------------------------------------------------------------

def format_search_space(param_name: str, space_def: Union[list, dict]) -> str:
    """
    Format the search space definition of the hyperparameters for the LaTeX table.
    
    args:
        param_name: name of the hyperparameter
        space_def: the search space definition according to train-gnn.yml

    returns:
        str: the formatted search space for in the table
    """
    if isinstance(space_def, list):
        # Discrete list of values
        values_str = ", ".join(str(v) for v in space_def)
        return f"$\\{{ {values_str} \\}}$"
    elif isinstance(space_def, dict) and "min" in space_def and "max" in space_def:
        # Continuous range
        min_val = space_def["min"]
        max_val = space_def["max"]
        if param_name == "lr":
            # Convert log values to exponential notation
            return f"$[10^{{{min_val}}}, 10^{{{max_val}}}]$"
        else:
            return f"$[{min_val}, {max_val}]$"
    else:
        return str(space_def)

def write_hyperparameter_table():
    """
    Write a .tex file comparing hyperparameter search space and optimal values in
    a well-formatted LaTeX table.
    """
    config_path = "src/config/gnn/train-gnn.yml"
    hyperparams_path = "results/gnn/hyperparameters.json"
    
    try:
        config = load_config(config_path)
        with open(hyperparams_path, 'r') as f:
            optimal_params = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        return
    except Exception as e:
        print(f"Error loading hyperparameter files: {e}")
        return
    
    # Create table header
    header = r"""\begin{table}[h!]
    \centering
    \caption{GNN hyperparameter grid and optimal values.}
    \label{tab: hyperparameter results}
    \begin{tabular}{l l l}
    \toprule
         parameter & search space & optimal value  \\
     \midrule"""
    
    footer = r"""    \bottomrule
    \end{tabular}
\end{table}"""
    
    lines = [header]
    
    # Define parameter mappings and order exactly as required
    param_mappings = [
        ("batch_size", "batch size", None),
        ("dropout", "dropout", None),
        ("epochs", "epochs", None),
        ("final_mlp_layers", "final mlp layers", "($M$)"),
        ("hidden_dim", "hidden dim", "($d$)"),
        ("lr", "learning rate", None),
        ("mlp_layers_per_gin", "mlp layers per gin", "($L_k$)"),
        ("num_gin_layers", "num gin layers", "($K$)"),
        ("readout", "readout", None)
    ]
    
    hyperopt_space = config.get("hyperopt_space", {})
    
    for param_key, display_name, symbol in param_mappings:
        if param_key in hyperopt_space and param_key in optimal_params:
            param_formatted = f"\\texttt{{{display_name}}}"
            if symbol:
                param_formatted += f" {symbol}"
            
            space_def = hyperopt_space[param_key]
            if isinstance(space_def, list):
                if param_key == "readout":
                    # Special formatting for readout
                    values_str = ", ".join([f"\\operatorname{{{v}}}" for v in space_def])
                    search_space = f"$\\{{{values_str} \\}}$"
                else:
                    # Regular discrete list
                    values_str = ", ".join(str(v) for v in space_def)
                    search_space = f"$\\{{ {values_str} \\}}$"
            elif isinstance(space_def, dict) and "min" in space_def and "max" in space_def:
                # Continuous range
                min_val = space_def["min"]
                max_val = space_def["max"]
                if param_key == "lr":
                    search_space = f"$[10^{{{min_val}}}, 10^{{{max_val}}}]$"
                else:
                    search_space = f"$[{min_val}, {max_val}]$"
            
            optimal_val = optimal_params[param_key]
            if param_key == "lr":
                lr_in_thousands = optimal_val * 1000
                optimal_formatted = f"${lr_in_thousands:.2f} \\cdot 10^{{-3}}$"
            elif isinstance(optimal_val, float):
                optimal_formatted = f"{optimal_val:.4f}"
            else:
                optimal_formatted = str(optimal_val)
            
            line = f"        {param_formatted} & {search_space} & {optimal_formatted} \\\\"
            lines.append(line)
    
    lines.append(footer)
    
    # Write to file
    output_path = "results/gnn/hyperparameters.tex"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"Wrote hyperparameter table to '{output_path}'")

write_hyperparameter_table()