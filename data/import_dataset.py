import pandas as pd
import sqlite3
import glob
import os

def import_excel_to_db():
    # Get the correct database path from config
    from config import DATABASE_PATH
    conn = sqlite3.connect(DATABASE_PATH)

    # Get the dataset directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    
    # Find all xlsx files in dataset folder
    files = glob.glob(os.path.join(dataset_dir, "*.xlsx"))

    for file in files:
        print(f"Processing {file}")

        try:
            df = pd.read_excel(file)

            # Ambil hanya kolom yang dibutuhkan (flexible column names)
            # Try common column names
            date_col = None
            price_col = None
            
            for col in df.columns:
                if col.lower() in ['date', 'tanggal']:
                    date_col = col
                if col.lower() in ['price', 'harga', 'price (rp)', 'pricerp']:
                    price_col = col
            
            if date_col is None or price_col is None:
                print(f"Skipping {file} - required columns not found")
                continue
                
            df = df[[date_col, price_col]]

            # Rename agar konsisten dengan database
            df.columns = ['date', 'price']

            # Format tanggal
            df['date'] = pd.to_datetime(df['date'], dayfirst=True)

            # Bersihkan format harga (hapus koma dan titik)
            df['price'] = (
                df['price']
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.replace('.', '', regex=False)
                .astype(float)
            )

            # Filter harga tidak valid
            df = df[df['price'] > 0]

            # Masukkan ke database
            df.to_sql("price_data", conn, if_exists="append", index=False)

            print(f"{file} imported successfully.")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

    conn.close()
    print("All datasets imported successfully.")

if __name__ == "__main__":
    import_excel_to_db()
