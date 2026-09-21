import pandas as pd


# =========================================================
# LOAD RESULTS
# =========================================================

input_file = "results/experiment_results.csv"

df = pd.read_csv(input_file)

print("\n" + "=" * 80)
print("              CLASSICAL RTI BENCHMARK ANALYSIS")
print("=" * 80)

print("\nTotal rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# OVERALL METHOD SUMMARY
# =========================================================

method_summary = (
    df.groupby("Method")[
        ["RMSE", "PSNR", "SSIM", "FSIM"]
    ]
    .mean()
    .reset_index()
)

print("\n")
print("=" * 80)
print("OVERALL METHOD AVERAGE")
print("=" * 80)

print(
    method_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =========================================================
# SENSOR COUNT SUMMARY
# =========================================================

sensor_summary = (
    df.groupby(
        ["Sensors", "Method"]
    )[
        ["RMSE", "PSNR", "SSIM", "FSIM"]
    ]
    .mean()
    .reset_index()
)

print("\n")
print("=" * 80)
print("SENSOR COUNT SUMMARY")
print("=" * 80)

print(
    sensor_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =========================================================
# TARGET TYPE SUMMARY
# =========================================================

target_type_summary = (
    df.groupby(
        ["Target_Type", "Method"]
    )[
        ["RMSE", "PSNR", "SSIM", "FSIM"]
    ]
    .mean()
    .reset_index()
)

print("\n")
print("=" * 80)
print("TARGET TYPE SUMMARY")
print("=" * 80)

print(
    target_type_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =========================================================
# TARGET COUNT SUMMARY
# =========================================================

target_count_summary = (
    df.groupby(
        ["Num_Targets", "Method"]
    )[
        ["RMSE", "PSNR", "SSIM", "FSIM"]
    ]
    .mean()
    .reset_index()
)

print("\n")
print("=" * 80)
print("NUMBER OF TARGETS SUMMARY")
print("=" * 80)

print(
    target_count_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =========================================================
# FULL CONDITION SUMMARY
# =========================================================

full_summary = (
    df.groupby(
        [
            "Sensors",
            "Target_Type",
            "Num_Targets",
            "Method"
        ]
    )[
        ["RMSE", "PSNR", "SSIM", "FSIM"]
    ]
    .mean()
    .reset_index()
)


# =========================================================
# SAVE SUMMARY
# =========================================================

output_file = (
    "results/benchmark_summary.csv"
)

full_summary.to_csv(
    output_file,
    index=False
)

print("\n")
print("=" * 80)
print("SUMMARY SAVED")
print("=" * 80)

print(
    output_file
)

print("\nAnalysis completed successfully.")