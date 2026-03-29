import os
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

class StockDatasetPreparation:
    """Prepares stock data from individual CSV files for RL training"""

    def __init__(self, data_dir='./stock_data', output_dir='./processed_data'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_all_stock_files(self):
        """Load all stock CSV files from data directory"""
        stock_files = list(Path(self.data_dir).glob('*.ns.csv'))
        print(f"Found {len(stock_files)} stock CSV files")

        all_data = []
        for file_path in stock_files:
            try:
                df = pd.read_csv(file_path)
                df['Stock'] = file_path.stem.replace('.ns', '')
                all_data.append(df)
                print(f"✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {e}")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        return None

    def calculate_technical_indicators(self, df):
        """Calculate RSI, ATR, and other technical indicators"""
        df = df.sort_values(['Stock', 'Date']).reset_index(drop=True)
        indicators = []

        for stock in df['Stock'].unique():
            stock_df = df[df['Stock'] == stock].copy()
            stock_df = stock_df.reset_index(drop=True)

            # RSI (14 periods)
            rsi_14 = self._calculate_rsi(stock_df['Close'].values, period=14)
            stock_df['RSI_14'] = rsi_14

            # ATR (14 periods)
            atr_14 = self._calculate_atr(
                stock_df['High'].values,
                stock_df['Low'].values,
                stock_df['Close'].values,
                period=14
            )
            stock_df['ATR_14'] = atr_14

            # Price Change - FIX: Use fill_method=None to avoid FutureWarning
            stock_df['Price_Change'] = stock_df['Close'].pct_change(fill_method=None)

            # Volatility (20-period) - FIX: Use fill_method=None
            stock_df['Volatility'] = stock_df['Close'].pct_change(fill_method=None).rolling(window=20).std()

            # P/E Ratio normalization
            stock_df['PE_Ratio'] = stock_df['Close'] / (stock_df['Close'].rolling(window=20).mean() + 1e-6)

            # Dividend yield
            stock_df['Dividend'] = stock_df['Dividends'].fillna(0) if 'Dividends' in stock_df.columns else 0

            indicators.append(stock_df)

        result_df = pd.concat(indicators, ignore_index=True)
        return result_df

    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / (down + 1e-6)

        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period

            rs = up / (down + 1e-6)
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi

    def _calculate_atr(self, highs, lows, closes, period=14):
        """Calculate Average True Range"""
        tr = np.zeros(len(highs))
        tr[0] = highs[0] - lows[0]

        for i in range(1, len(highs)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )

        atr = np.convolve(tr, np.ones(period)/period, mode='valid')
        atr = np.concatenate([np.full(period-1, atr[0]), atr])

        return atr

    def normalize_features(self, df):
        """Normalize features to 0-1 range"""
        df_normalized = df.copy()

        # Price normalization (0-1)
        price_min, price_max = df['Close'].min(), df['Close'].max()
        df_normalized['Price_Norm'] = (df['Close'] - price_min) / (price_max - price_min + 1e-6)

        # Volatility normalization (0-1)
        vol_max = df['Volatility'].max()
        df_normalized['Volatility_Norm'] = np.minimum(df['Volatility'] / (vol_max + 1e-6), 1.0)

        # P/E Ratio normalization (0-1)
        pe_min, pe_max = df['PE_Ratio'].min(), df['PE_Ratio'].max()
        df_normalized['PE_Ratio_Norm'] = (df['PE_Ratio'] - pe_min) / (pe_max - pe_min + 1e-6)

        # Dividend normalization (0-1)
        div_max = df['Dividend'].max()
        df_normalized['Dividend_Norm'] = np.minimum(df['Dividend'] / (div_max + 1e-6), 1.0)

        # RSI normalization (0-1)
        df_normalized['RSI_Norm'] = df['RSI_14'] / 100.0

        # ATR normalization (0-1)
        atr_max = df['ATR_14'].max()
        df_normalized['ATR_Norm'] = np.minimum(df['ATR_14'] / (atr_max + 1e-6), 1.0)

        # Price Change normalization (-1 to 1)
        pc_max = max(abs(df['Price_Change'].min()), abs(df['Price_Change'].max()))
        df_normalized['Price_Change_Norm'] = df['Price_Change'] / (pc_max + 1e-6)
        df_normalized['Price_Change_Norm'] = df_normalized['Price_Change_Norm'].clip(-1, 1)

        return df_normalized

    def prepare_training_data(self):
        """Main function to prepare complete dataset"""
        print("\n" + "="*70)
        print("DATASET PREPARATION STARTED")
        print("="*70)

        # Load all stocks
        combined_df = self.load_all_stock_files()
        if combined_df is None:
            print("\n❌ No data found. CSV files should be in ./stock_data/")
            return None

        print(f"\n✓ Total records loaded: {len(combined_df)}")
        print(f"✓ Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")

        # Calculate indicators
        print("\nCalculating technical indicators...")
        df_with_indicators = self.calculate_technical_indicators(combined_df)

        # Normalize features
        print("Normalizing features to 0-1 range...")
        df_normalized = self.normalize_features(df_with_indicators)

        # Select relevant columns
        feature_columns = [
            'Stock', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Price_Norm', 'Volatility_Norm', 'PE_Ratio_Norm',
            'Dividend_Norm', 'RSI_Norm', 'ATR_Norm', 'Price_Change_Norm'
        ]

        available_columns = [col for col in feature_columns if col in df_normalized.columns]
        rl_dataset = df_normalized[available_columns].dropna()

        print(f"\n✓ Final dataset shape: {rl_dataset.shape}")
        print(f"✓ Features ready: {len(available_columns)} columns")

        # Save dataset
        output_file = os.path.join(self.output_dir, 'rl_training_dataset.csv')
        rl_dataset.to_csv(output_file, index=False)
        print(f"\n✓ Dataset saved: {output_file}")

        print("\n" + "="*70)
        print("✅ DATASET PREPARATION COMPLETE!")
        print("="*70 + "\n")

        return rl_dataset

if __name__ == "__main__":
    prep = StockDatasetPreparation(data_dir='./stock_data', output_dir='./processed_data')
    dataset = prep.prepare_training_data()

    if dataset is not None:
        print("Dataset Preview (first 5 rows):")
        print(dataset.head())
        print("\nDataset Columns:")
        print(dataset.columns.tolist())
