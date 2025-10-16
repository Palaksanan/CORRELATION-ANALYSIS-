
import pandas as pd
import numpy as np

# Load the Excel file, skipping messy header rows
df = pd.read_excel("E:/p/BASIC STATISTICS LAB/NEW/correlation analysis.xlsx", skiprows=8, header=None)

# Safely extract numeric columns by position
# Column 4: Electricity consumption (X1)
# Column 5: House size (X2)
# Column 6: AC usage hours (X3)
X1 = pd.to_numeric(df.iloc[:, 4], errors='coerce').dropna().values
X2 = pd.to_numeric(df.iloc[:, 5], errors='coerce').dropna().values
X3 = pd.to_numeric(df.iloc[:, 6], errors='coerce').dropna().values

# Ensure all arrays are the same length
min_len = min(len(X1), len(X2), len(X3))
X1, X2, X3 = X1[:min_len], X2[:min_len], X3[:min_len]

# --- Pearson's Correlation Coefficients ---
r12 = np.corrcoef(X1, X2)[0, 1]
r13 = np.corrcoef(X1, X3)[0, 1]
r23 = np.corrcoef(X2, X3)[0, 1]

print("\n--- Pearson's Correlation Coefficients ---")
print(f"r12 (Electricity vs Area): {r12:.4f}")
print(f"r13 (Electricity vs AC Hours): {r13:.4f}")
print(f"r23 (Area vs AC Hours): {r23:.4f}")

# --- Multiple Correlation Coefficients ---
R1_23 = np.sqrt((r12**2 + r13**2 - 2*r12*r13*r23) / (1 - r23**2))
R2_13 = np.sqrt((r12**2 + r23**2 - 2*r12*r23*r13) / (1 - r13**2))
R3_12 = np.sqrt((r13**2 + r23**2 - 2*r13*r23*r12) / (1 - r12**2))

print("\n--- Multiple Correlation Coefficients ---")
print(f"R1.23: {R1_23:.4f}")
print(f"R2.13: {R2_13:.4f}")
print(f"R3.12: {R3_12:.4f}")

# --- Partial Correlation Coefficients ---
def partial_corr(r_xy, r_xz, r_yz):
    return (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

r12_3 = partial_corr(r12, r13, r23)
r13_2 = partial_corr(r13, r12, r23)
r23_1 = partial_corr(r23, r12, r13)

print("\n--- Partial Correlation Coefficients ---")
print(f"r12.3: {r12_3:.4f}")
print(f"r13.2: {r13_2:.4f}")
print(f"r23.1: {r23_1:.4f}")
