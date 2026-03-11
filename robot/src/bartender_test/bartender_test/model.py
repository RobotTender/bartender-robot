#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

def main():
    csv_file = 'pour_data.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run 'ros2 run bartender_test monitor pour' and trigger some captures first.")
        return

    df = pd.read_csv(csv_file)
    if df.empty:
        print("Error: pour_data.csv is empty.")
        return

    print("Current Captured Data:")
    print(df)

    # Ensure 'weight' column exists
    if 'weight' not in df.columns:
        df['weight'] = np.nan

    # Prompt only for rows where weight is missing
    missing_weights = df['weight'].isna()
    if missing_weights.any():
        print(f"\nFound {missing_weights.sum()} new captures missing weights.")
        for i in df[missing_weights].index:
            row = df.loc[i]
            while True:
                try:
                    val = input(f"[{i+1}/{len(df)}] Enter bottle weight (g) for capture at {row['rx']:.2f}° (or 's' to skip/delete): ")
                    if val.lower() == 's':
                        df.at[i, 'weight'] = -1 # Mark for deletion
                        break
                    w = float(val)
                    df.at[i, 'weight'] = w
                    break
                except ValueError:
                    print("Invalid input. Enter a number or 's'.")
        
        # Clean up: Remove skipped/deleted rows
        df = df[df['weight'] > 0]
        df.to_csv(csv_file, index=False)
        print(f"Updated {csv_file} with new data.")

    if len(df) < 2:
        print("Need at least 2 valid data points to build a model.")
        return

    # Modeling: Weight vs Ry (Tilt Angle)
    # Typically, the tilt angle where the first drop occurs is related to the volume remaining.
    # Higher weight (more full) -> Smaller tilt angle to start pouring.
    # Lower weight (less full) -> Larger tilt angle to start pouring.
    
    X = df[['weight']].values
    y = df['ry'].values # Using Ry as the primary tilt axis for this dataset

    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)

    print("\n" + "="*40)
    print("FLOW MODEL Q(alpha) - Calibration Results")
    print(f"Relationship: Ry = {slope:.4f} * Weight + {intercept:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    print("="*40)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X), color='red', label='Linear Fit')
    plt.xlabel('Bottle Weight (g)')
    plt.ylabel('Tilt Angle Ry (deg)')
    plt.title('First Drop Calibration: Weight vs. Tilt Angle')
    plt.legend()
    plt.grid(True)
    
    plot_file = 'calibration_plot.png'
    plt.savefig(plot_file)
    print(f"Saved calibration plot to {plot_file}")
    
    # Predict function
    print("\nModel for use in Phase 4:")
    print(f"def get_expected_angle(weight_g):")
    print(f"    return {slope:.4f} * weight_g + {intercept:.4f}")

if __name__ == '__main__':
    main()
