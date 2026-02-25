
import pandas as pd
import sys
import os

# Add parent directory to path to import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stock_analysis.support_resistance_analyzer as analyzer

def inspect_data():
    symbol = "002165"
    df = analyzer.get_stock_data(symbol)
    
    if df is None:
        print("Data not found")
        return

    # Check data around 2025-12-17 to 2026-01-19
    start_date = "2025-12-01"
    end_date = "2026-01-20"
    
    mask = (df.index >= start_date) & (df.index <= end_date)
    subset = df.loc[mask]
    
    print(f"Data for {symbol} from {start_date} to {end_date}:")
    for date, row in subset.iterrows():
        # Check if volume is 'huge' (e.g. > 2x average)
        # We need a rolling average for volume
        pass 
        
    # Calculate 20-day MA for volume to check "huge volume"
    subset['MA_Vol_20'] = subset['volume'].rolling(window=20).mean()
    
    for date, row in subset.iterrows():
        vol_ratio = row['volume'] / row['MA_Vol_20'] if pd.notna(row['MA_Vol_20']) and row['MA_Vol_20'] > 0 else 0
        print(f"{date.strftime('%Y-%m-%d')}: Open={row['open']:.2f}, High={row['high']:.2f}, Low={row['low']:.2f}, Close={row['close']:.2f}, Vol={row['volume']:.0f} (Ratio: {vol_ratio:.2f})")

if __name__ == "__main__":
    inspect_data()
