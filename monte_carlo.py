import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from common import State


def sample_initial_conditions(num_runs: int, seed: int, planet_params: dict, rocket_params: dict) -> list:
    """
    Draws randomized initial states for all runs up front, so they don't depend on how runs are split across workers.
    Both guidance methods use this, so the same seed gives them identical initial conditions.
    """

    all_initial_conditions = []
    np.random.seed(seed)
    for i in range(num_runs):
        initial_conditions = np.zeros(7)
        initial_conditions[State.X] = np.random.uniform(-50000.0, 50000.0) # m
        initial_conditions[State.Y] = np.random.uniform(-50000.0, 50000.0) # m
        initial_conditions[State.Z] = planet_params['R'] + np.random.uniform(50000.0, 150000.0) # m
        initial_conditions[State.VX] = np.random.uniform(-1000.0, 1000.0) # m/s
        initial_conditions[State.VY] = np.random.uniform(-1000.0, 1000.0) # m/s
        initial_conditions[State.VZ] = np.random.uniform(-500.0, 500.0) # m/s
        initial_conditions[State.M] = rocket_params['mass_start'] # kg
        all_initial_conditions.append(initial_conditions)

    return all_initial_conditions


def _silence_stdout():
    """
    Pool initializer: redirects file descriptor 1 to /dev/null in the worker process only.
    Done at the file-descriptor level so both Python prints and C/C++ output (e.g. IPOPT logs) are silenced.
    stderr is left untouched so tracebacks and warnings remain visible.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)


def _safe_call(run_func: callable, initial_conditions):
    """
    Calls run_func in a worker, converting any exception into a failed result so a single bad run cannot abort the batch.
    """
    try:
        return run_func(initial_conditions)
    except Exception as e:
        return {'success': False, 'error': repr(e)}


def _summarize_result(result: dict) -> str:
    if result.get('success'):
        return f"OK  tf={result['tf']:.2f} s  m_f={result['X'][State.M, -1]:.2f} kg  solve={result['solve_time']:.2f} s"
    if 'error' in result:
        return f"FAILED ({result['error']})"
    return "FAILED"


def run_monte_carlo(run_func: callable, initial_conditions_list: list, num_workers: int) -> list:
    """
    Runs one trajectory generation per initial condition, either serially or across a pool of worker processes.

    Args:
        run_func (callable): Function taking a single initial-conditions array and returning a result dict.
            Must be picklable when num_workers > 1 (a module-level function or a functools.partial of one).
        initial_conditions_list (list): Initial-condition arrays, one per run
        num_workers (int): Number of worker processes. 1 runs serially in this process with full verbose output.

    Returns:
        results (list): Result dicts in the same order as initial_conditions_list
    """

    num_runs = len(initial_conditions_list)

    # Serial: run each initial condition one after another in this process
    if num_workers is None or num_workers <= 1 or num_runs <= 1:
        results = []
        for i in range(num_runs):
            initial_conditions = initial_conditions_list[i]
            print(f"--- Starting run {i+1}/{num_runs} ---")
            print(f"Initial conditions (run {i+1}): {initial_conditions}")
            result = run_func(initial_conditions)
            results.append(result)
        return results

    # Parallel: hand every run to a pool of worker processes
    num_workers = min(num_workers, num_runs)
    print(f"Running {num_runs} runs in parallel on {num_workers} worker processes (per-run output suppressed; set num_workers = 1 for verbose serial output)")

    # Runs finish out of order, so each result is stored at its run index rather than appended
    results = [None] * num_runs
    with ProcessPoolExecutor(max_workers=num_workers, initializer=_silence_stdout) as executor:
        # Submit all runs. Each submit returns a future (a handle to the pending result),
        # so remember which run index each future belongs to.
        run_index_of_future = {}
        for i in range(num_runs):
            future = executor.submit(_safe_call, run_func, initial_conditions_list[i])
            run_index_of_future[future] = i

        # as_completed yields each future as soon as its run finishes, in completion order
        num_done = 0
        for future in as_completed(run_index_of_future):
            i = run_index_of_future[future]
            try:
                results[i] = future.result()
            except BrokenProcessPool as e:
                raise RuntimeError(f"A worker process was killed abruptly while running run {i+1} (most often the OS out-of-memory killer). "
                                   f"Reduce num_workers.") from e
            num_done += 1
            print(f"[{num_done}/{num_runs}] run {i+1}: {_summarize_result(results[i])}")

    return results


def print_summary(results: list, initial_conditions_list: list) -> list:
    """
    Prints the overall success rate and the initial conditions of any failed runs.
    Returns the successful result dicts, in run order.
    """

    successful_runs = []
    failed_initial_conditions = []
    for i in range(len(results)):
        if results[i]["success"]:
            successful_runs.append(results[i])
        else:
            failed_initial_conditions.append(initial_conditions_list[i])
    num_runs = len(results)
    num_success = len(successful_runs)

    print("\n--- Overall Summary ---")
    print(f"Total runs attempted: {num_runs}")
    print(f"Successful runs: {num_success} ({num_success/num_runs*100:.1f}%)")

    if failed_initial_conditions:
        print("\n--- Failed Initial Conditions ---")
        for initial_condition in failed_initial_conditions:
            print(initial_condition)

    return successful_runs


def save_monte_carlo_results(file_path, results, initial_conditions, settings):
    """
    Save Monte Carlo results (summary statistics plus one line per run) to a JSON file. Keeping the file under version
    control makes regressions visible with "git diff" after re-running the Monte Carlo on changed code. Solve times are
    left out on purpose, since they vary between identical runs.

    Args:
        file_path (str): Output file path (its directory is created if needed)
        results (list): Result dicts returned by the trajectory generator, one per run
        initial_conditions (list): Initial state vectors, one per run, in the same order as results
        settings (dict): Settings that produced the results (e.g. num_runs, seed, N, etc.), saved for reference
    """

    num_digits = 3

    # Per-run records
    runs = []
    num_runs = len(results)
    for i in range(num_runs):
        # A failed run may not have an iteration count (e.g. a failed Hermite-Simpson run returns just {'success': False})
        if 'iterations' in results[i]:
            iterations = int(results[i]['iterations'])
        else:
            iterations = None

        # A failed run may not have returned a trajectory at all
        if 'X' in results[i]:
            final_mass = _finite_or_none(results[i]['X'][State.M, -1], num_digits)
            tf = _finite_or_none(results[i]['tf'], num_digits)
        else:
            final_mass = None
            tf = None

        saved_initial_conditions = []
        for value in initial_conditions[i]:
            saved_initial_conditions.append(_finite_or_none(value, num_digits))

        # Each run is saved to file as a dict with the run number, success flag, iteration count, final mass, final time, and initial conditions
        run = {
            'run': i + 1,
            'success': bool(results[i]['success']),
            'iterations': iterations,
            'final_mass': final_mass,
            'tf': tf,
            'initial_conditions': saved_initial_conditions,
        }
        runs.append(run)

    # Summary statistics over the successful runs only
    successful_iterations = []
    successful_final_masses = []
    successful_tfs = []
    for i in range(num_runs):
        if not results[i]['success']:
            continue
        successful_iterations.append(results[i]['iterations'])
        successful_final_masses.append(results[i]['X'][State.M, -1])
        successful_tfs.append(results[i]['tf'])

    summary = {
        'num_runs': num_runs,
        'num_success': len(successful_tfs),
        'iterations': _statistics(successful_iterations, num_digits),
        'final_mass': _statistics(successful_final_masses, num_digits),
        'tf': _statistics(successful_tfs, num_digits),
    }

    # Settings and summary are pretty-printed.
    settings_json = _json_dumps_nested(settings)
    summary_json = _json_dumps_nested(summary)

    run_lines = []
    for run in runs:
        run_lines.append('    ' + json.dumps(run))
    runs_json = ',\n'.join(run_lines)

    lines = []
    lines.append('{')
    lines.append('  "settings": ' + settings_json + ',')
    lines.append('  "summary": ' + summary_json + ',')
    lines.append('  "runs": [')
    lines.append(runs_json)
    lines.append('  ]')
    lines.append('}')
    file_contents = '\n'.join(lines) + '\n'

    # Create the output directory if it doesn't exist yet
    directory = os.path.dirname(file_path)
    if directory != '':
        os.makedirs(directory, exist_ok=True)

    with open(file_path, 'w') as f:
        f.write(file_contents)


def _finite_or_none(value, num_digits):
    """Rounds a number for saving, mapping NaN/inf to None so the file stays valid JSON."""

    if value is None or not np.isfinite(value):
        return None
    return round(float(value), num_digits)


def _statistics(values, num_digits):
    if len(values) == 0:
        return None
    values = np.asarray(values, dtype=float)
    return {
        'mean': _finite_or_none(np.mean(values), num_digits),
        'std': _finite_or_none(np.std(values), num_digits),
        'median': _finite_or_none(np.median(values), num_digits),
        'p99': _finite_or_none(np.percentile(values, 99), num_digits),
        'min': _finite_or_none(np.min(values), num_digits),
        'max': _finite_or_none(np.max(values), num_digits),
    }


def _numpy_to_python(value):
    """Converts NumPy scalars/arrays into plain Python ints, floats and lists, which json.dumps can serialize."""

    return value.tolist()


def _json_dumps_nested(obj):
    """
    Pretty-prints obj as JSON with a 2-space indent, then indents every line after the first by a further 2 spaces,
    so the result lines up when placed as a value inside the top-level JSON object.
    """

    text = json.dumps(obj, indent=2, default=_numpy_to_python)
    text = text.replace('\n', '\n  ')
    return text
