# this is data preprocessing + target/label creation.
import pandas as pd
import numpy as np

def prepare_agrivault_dataset(input_csv_path, output_csv_path):
    print("Loading raw dataset...")
    df = pd.read_csv(input_csv_path)

    # 1. Filter for Onions in Maharashtra - Nashik District
    onion_df = df[
        (df['Commodity'].str.lower() == 'onion') & 
        (df['STATE'].str.lower() == 'maharashtra') & 
        (df['District Name'].str.lower() == 'nashik')
    ].copy()

    # 2. Convert Price Date to datetime and sort by Market + Date
    onion_df['Price Date'] = pd.to_datetime(onion_df['Price Date'], format='mixed')
    onion_df['Price_Per_Kg'] = onion_df['Modal_Price'] / 100.0 # because modal price is represented in Rs per quintal so, 1 quintal = 100 kg, ₹2500 per quintal, ₹2500 / 100 = ₹25/kg, Price_Per_Kg
    onion_df = onion_df.sort_values(['Market Name', 'Price Date']).reset_index(drop=True)  # without drop=True we get a new column for the new index so insted we wanted to remove the old index and keep the new one in place of old one.

    # 3. Shift 30 rows back and 60 rows forward PER MARKET
    onion_df['Price_30D_Lag'] = onion_df.groupby('Market Name')['Price_Per_Kg'].shift(30) # Give each row the value that was 30 rows before it. example shift(2) that becomes or leaves the first two rows and next start with the 3rd row.
    onion_df['Price_Change_30D_Pct'] = (
        (onion_df['Price_Per_Kg'] - onion_df['Price_30D_Lag']) / onion_df['Price_30D_Lag'] # here we are calculating the percentage of the price change from past 30 rows 
    )
    onion_df['Future_Price_60D'] = onion_df.groupby('Market Name')['Price_Per_Kg'].shift(-60)

    # Drop NaN values created by market shifts
    onion_df = onion_df.dropna().reset_index(drop=True)

    # 4. Financial Decision Math
    monthly_storage_cost_per_kg = 2.0  # Rs 2/kg/month -> Rs 4/kg for 60 days
    spoilage_rate_60d = 0.08           # 8% crop spoilage over 60 days
    quantity_kg = 100.0

    rev_sell_now = quantity_kg * onion_df['Price_Per_Kg']
    surviving_kg = quantity_kg * (1 - spoilage_rate_60d)
    rev_stored = surviving_kg * onion_df['Future_Price_60D']
    cost_storage = quantity_kg * (monthly_storage_cost_per_kg * 2)
    net_profit_storing = rev_stored - cost_storage # here we get the net profit if we store the onion and removing the storage cost form the revenue

    # This is the most important lines
    # 5. Target Labeling (1 = Store, 0 = Sell)
    onion_df['Target_Should_Store'] = (net_profit_storing >= (rev_sell_now * 1.10)).astype(int)  # 1.10 means 
    # This means storing must generate at least 10% more revenue than selling now.

    # .astype(int) means Before .astype(int), the result is Boolean means True, False so ML mode commonly use numerical labels.
    # It converts the True  → 1 False → 0 
    # 1 = Store 0 = Sell

    onion_df.to_csv(output_csv_path, index=False)
    print(f"Data processing complete! Saved to {output_csv_path}")
    print(f"Total processed samples: {len(onion_df)}")
    print(f"Class Distribution:\n{onion_df['Target_Should_Store'].value_counts()}") # Class distribution is done here which tells us that how many samples are 0 and how many samples are 1 with this we can know check that whether your target classes are reasonably balanced.

if __name__ == '__main__':
    prepare_agrivault_dataset('data/Agriculture_price_dataset.csv', 'data/processed_onion.csv')