import os
import csv
import numpy as np


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = "results"

INPUT_FILE = os.path.join(
    RESULTS_DIR,
    "ce_benchmark_results.csv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "ce_analysis_summary.csv"
)


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"\nCould not find:\n{INPUT_FILE}\n\n"
            "Run the benchmark first:\n"
            "python src/benchmark_ce.py"
        )

    with open(INPUT_FILE, "r", newline="") as file:

        reader = csv.DictReader(file)

        results = []

        for row in reader:

            results.append({
                "Sensors": int(row["Sensors"]),
                "Links": int(row["Links"]),
                "Target_Type": row["Target_Type"],
                "Target_Count": int(row["Target_Count"]),
                "Method": row["Method"],
                "RMSE": float(row["RMSE"]),
                "PSNR": float(row["PSNR"]),
                "SSIM": float(row["SSIM"]),
                "FSIM": float(row["FSIM"])
            })

    return results


# ============================================================
# PRINT ALL RESULTS
# ============================================================

def print_all_results(results):

    print()
    print("=" * 100)
    print("CE + CNN BENCHMARK RESULTS")
    print("=" * 100)

    print(
        f"{'Sensors':<9}"
        f"{'Type':<10}"
        f"{'Targets':<9}"
        f"{'Method':<12}"
        f"{'RMSE':<10}"
        f"{'PSNR':<10}"
        f"{'SSIM':<10}"
        f"{'FSIM':<10}"
    )

    print("-" * 100)

    for row in results:

        print(
            f"{row['Sensors']:<9}"
            f"{row['Target_Type']:<10}"
            f"{row['Target_Count']:<9}"
            f"{row['Method']:<12}"
            f"{row['RMSE']:<10.4f}"
            f"{row['PSNR']:<10.4f}"
            f"{row['SSIM']:<10.4f}"
            f"{row['FSIM']:<10.4f}"
        )


# ============================================================
# GROUP RESULTS
# ============================================================

def group_results(results):

    groups = {}

    for row in results:

        key = (
            row["Sensors"],
            row["Target_Type"],
            row["Target_Count"]
        )

        if key not in groups:
            groups[key] = {}

        groups[key][row["Method"]] = row

    return groups


# ============================================================
# CALCULATE IMPROVEMENT
# ============================================================

def calculate_improvement(tikhonov, ce):

    rmse_improvement = (
        (tikhonov["RMSE"] - ce["RMSE"])
        / tikhonov["RMSE"]
    ) * 100

    psnr_improvement = (
        (ce["PSNR"] - tikhonov["PSNR"])
        / abs(tikhonov["PSNR"])
    ) * 100

    ssim_improvement = (
        (ce["SSIM"] - tikhonov["SSIM"])
        / max(abs(tikhonov["SSIM"]), 1e-12)
    ) * 100

    fsim_improvement = (
        (ce["FSIM"] - tikhonov["FSIM"])
        / max(abs(tikhonov["FSIM"]), 1e-12)
    ) * 100

    return {
        "RMSE_Improvement_%": rmse_improvement,
        "PSNR_Improvement_%": psnr_improvement,
        "SSIM_Improvement_%": ssim_improvement,
        "FSIM_Improvement_%": fsim_improvement
    }


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

def create_comparison_table(results):

    groups = group_results(results)

    comparison = []

    for key in sorted(groups.keys()):

        sensors, target_type, target_count = key

        methods = groups[key]

        if "Tikhonov" not in methods:
            continue

        if "CE+CNN" not in methods:
            continue

        tikhonov = methods["Tikhonov"]
        ce = methods["CE+CNN"]

        improvement = calculate_improvement(
            tikhonov,
            ce
        )

        comparison.append({

            "Sensors": sensors,

            "Target_Type": target_type,

            "Target_Count": target_count,

            "Tikhonov_RMSE":
                tikhonov["RMSE"],

            "CE_CNN_RMSE":
                ce["RMSE"],

            "RMSE_Improvement_%":
                improvement["RMSE_Improvement_%"],

            "Tikhonov_PSNR":
                tikhonov["PSNR"],

            "CE_CNN_PSNR":
                ce["PSNR"],

            "PSNR_Improvement_%":
                improvement["PSNR_Improvement_%"],

            "Tikhonov_SSIM":
                tikhonov["SSIM"],

            "CE_CNN_SSIM":
                ce["SSIM"],

            "SSIM_Improvement_%":
                improvement["SSIM_Improvement_%"],

            "Tikhonov_FSIM":
                tikhonov["FSIM"],

            "CE_CNN_FSIM":
                ce["FSIM"],

            "FSIM_Improvement_%":
                improvement["FSIM_Improvement_%"]
        })

    return comparison


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison(comparison):

    print()
    print("=" * 120)
    print("TIKHONOV vs CE + CNN")
    print("=" * 120)

    print(
        f"{'Sensors':<9}"
        f"{'Type':<10}"
        f"{'Targets':<9}"
        f"{'RMSE Imp.%':<12}"
        f"{'PSNR Imp.%':<12}"
        f"{'SSIM Imp.%':<12}"
        f"{'FSIM Imp.%':<12}"
    )

    print("-" * 120)

    for row in comparison:

        print(
            f"{row['Sensors']:<9}"
            f"{row['Target_Type']:<10}"
            f"{row['Target_Count']:<9}"
            f"{row['RMSE_Improvement_%']:<12.2f}"
            f"{row['PSNR_Improvement_%']:<12.2f}"
            f"{row['SSIM_Improvement_%']:<12.2f}"
            f"{row['FSIM_Improvement_%']:<12.2f}"
        )


# ============================================================
# FIND BEST CASES
# ============================================================

def find_best_cases(results):

    ce_results = [
        row
        for row in results
        if row["Method"] == "CE+CNN"
    ]

    if not ce_results:
        return

    best_rmse = min(
        ce_results,
        key=lambda x: x["RMSE"]
    )

    best_psnr = max(
        ce_results,
        key=lambda x: x["PSNR"]
    )

    best_ssim = max(
        ce_results,
        key=lambda x: x["SSIM"]
    )

    best_fsim = max(
        ce_results,
        key=lambda x: x["FSIM"]
    )

    print()
    print("=" * 90)
    print("BEST CE + CNN RESULTS")
    print("=" * 90)

    print()
    print("Best RMSE:")
    print(best_rmse)

    print()
    print("Best PSNR:")
    print(best_psnr)

    print()
    print("Best SSIM:")
    print(best_ssim)

    print()
    print("Best FSIM:")
    print(best_fsim)


# ============================================================
# SENSOR COUNT ANALYSIS
# ============================================================

def sensor_analysis(results):

    print()
    print("=" * 90)
    print("SENSOR COUNT ANALYSIS — CE + CNN")
    print("=" * 90)

    ce_results = [
        row
        for row in results
        if row["Method"] == "CE+CNN"
    ]

    sensors = sorted(
        set(row["Sensors"] for row in ce_results)
    )

    print()
    print(
        f"{'Sensors':<10}"
        f"{'RMSE':<12}"
        f"{'PSNR':<12}"
        f"{'SSIM':<12}"
        f"{'FSIM':<12}"
    )

    print("-" * 60)

    for sensor_count in sensors:

        subset = [
            row
            for row in ce_results
            if row["Sensors"] == sensor_count
        ]

        if not subset:
            continue

        print(
            f"{sensor_count:<10}"
            f"{np.mean([x['RMSE'] for x in subset]):<12.4f}"
            f"{np.mean([x['PSNR'] for x in subset]):<12.4f}"
            f"{np.mean([x['SSIM'] for x in subset]):<12.4f}"
            f"{np.mean([x['FSIM'] for x in subset]):<12.4f}"
        )


# ============================================================
# TARGET COUNT ANALYSIS
# ============================================================

def target_count_analysis(results):

    print()
    print("=" * 90)
    print("TARGET COUNT ANALYSIS — CE + CNN")
    print("=" * 90)

    ce_results = [
        row
        for row in results
        if row["Method"] == "CE+CNN"
    ]

    target_counts = sorted(
        set(row["Target_Count"] for row in ce_results)
    )

    print()
    print(
        f"{'Targets':<10}"
        f"{'RMSE':<12}"
        f"{'PSNR':<12}"
        f"{'SSIM':<12}"
        f"{'FSIM':<12}"
    )

    print("-" * 60)

    for count in target_counts:

        subset = [
            row
            for row in ce_results
            if row["Target_Count"] == count
        ]

        print(
            f"{count:<10}"
            f"{np.mean([x['RMSE'] for x in subset]):<12.4f}"
            f"{np.mean([x['PSNR'] for x in subset]):<12.4f}"
            f"{np.mean([x['SSIM'] for x in subset]):<12.4f}"
            f"{np.mean([x['FSIM'] for x in subset]):<12.4f}"
        )


# ============================================================
# SOLID VS SMOOTH
# ============================================================

def target_type_analysis(results):

    print()
    print("=" * 90)
    print("SOLID vs SMOOTH — CE + CNN")
    print("=" * 90)

    ce_results = [
        row
        for row in results
        if row["Method"] == "CE+CNN"
    ]

    for target_type in ["solid", "smooth"]:

        subset = [
            row
            for row in ce_results
            if row["Target_Type"] == target_type
        ]

        if not subset:
            continue

        print()
        print(target_type.upper())

        print(
            f"RMSE : {np.mean([x['RMSE'] for x in subset]):.4f}"
        )

        print(
            f"PSNR : {np.mean([x['PSNR'] for x in subset]):.4f} dB"
        )

        print(
            f"SSIM : {np.mean([x['SSIM'] for x in subset]):.4f}"
        )

        print(
            f"FSIM : {np.mean([x['FSIM'] for x in subset]):.4f}"
        )


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

def save_comparison(comparison):

    fieldnames = list(comparison[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in comparison:
            writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print("RTI CE + CNN RESULT ANALYSIS")
    print("=" * 90)

    print()
    print("Input:")
    print(INPUT_FILE)

    results = load_results()

    print()
    print(
        f"Loaded {len(results)} reconstruction results."
    )

    print_all_results(results)

    comparison = create_comparison_table(
        results
    )

    print_comparison(
        comparison
    )

    sensor_analysis(
        results
    )

    target_count_analysis(
        results
    )

    target_type_analysis(
        results
    )

    find_best_cases(
        results
    )

    save_comparison(
        comparison
    )

    print()
    print("=" * 90)
    print("ANALYSIS COMPLETED")
    print("=" * 90)

    print()
    print(
        "Comparison saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()