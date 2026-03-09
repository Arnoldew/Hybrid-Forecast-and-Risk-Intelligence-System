import pandas as pd
import numpy as np
import os


def load_all_data():
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    
    # Build correct file paths - matching actual filenames in dataset folder
    path_2022 = os.path.join(dataset_dir, "1 Jan 2022 - 31 Dec 2022.xlsx")
    path_2023 = os.path.join(dataset_dir, "1 Jan 2023 - 31 Dec 2023.xlsx")
    path_2024 = os.path.join(dataset_dir, "1 Jan 2024 - 31 Dec 2024.xlsx")
    path_2025 = os.path.join(dataset_dir, "1 Jan 2025 - 31 Dec 2025.xlsx")

    # Check which files exist and load them
    available_paths = []
    for path in [path_2022, path_2023, path_2024, path_2025]:
        if os.path.exists(path):
            available_paths.append(path)
        else:
            print(f"Warning: File not found - {path}")
    
    if not available_paths:
        raise FileNotFoundError("No dataset files found in dataset folder!")
    
    dfs = []
    for path in available_paths:
        try:
            df = pd.read_excel(path)
            dfs.append(df)
            print(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    if not dfs:
        raise ValueError("No data could be loaded from dataset files")
    
    df_all = pd.concat(dfs, ignore_index=True)

    # Unify price column
    if 'Price (Rp)' in df_all.columns and 'Price(Rp)' in df_all.columns:
        df_all['Price'] = df_all['Price (Rp)'].fillna(df_all['Price(Rp)'])
    elif 'Price (Rp)' in df_all.columns:
        df_all['Price'] = df_all['Price (Rp)']
    elif 'Price(Rp)' in df_all.columns:
        df_all['Price'] = df_all['Price(Rp)']

    df_all = df_all.drop(columns=[col for col in ['Price (Rp)', 'Price(Rp)'] if col in df_all.columns])

    df_all['Date'] = pd.to_datetime(df_all['Date'])
    df_all['Price'] = pd.to_numeric(df_all['Price'], errors='coerce')

    df_all = df_all.sort_values('Date')
    df_all.set_index('Date', inplace=True)

    df_all = df_all[~df_all.index.duplicated(keep='first')]

    df_all.loc[df_all['Price'] <= 0, 'Price'] = np.nan
    df_all = df_all.dropna(subset=['Price'])

    # df_all = df_all.asfreq('D')
    # df_all['Price'] = df_all['Price'].interpolate(method='linear')

    return df_all