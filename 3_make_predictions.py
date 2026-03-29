import numpy as np
import pandas as pd
import joblib
import os
import pickle
from datetime import datetime


class RLPredictionEngine:
    """Enhanced prediction engine with stock selection and confidence analysis"""

    def __init__(self, model_dir='./rl_models', dataset_path='./processed_data/rl_training_dataset.csv'):
        self.model_dir = model_dir
        self.dataset_path = dataset_path
        self.model_strong = None
        self.model_weak = None
        self.scaler = None
        self.feature_columns = None
        self.available_stocks = []
        self.training_metrics = {}

        self.load_models()
        self.load_available_stocks()
        self.load_training_metrics()

    def load_models(self):
        """Load trained models"""
        print("\n" + "="*70)
        print("LOADING TRAINED MODELS")
        print("="*70)

        try:
            strong_path = os.path.join(self.model_dir, 'rl_model_strong_100.joblib')
            weak_path = os.path.join(self.model_dir, 'rl_model_weak_50.joblib')
            scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
            feature_path = os.path.join(self.model_dir, 'feature_columns.pkl')

            self.model_strong = joblib.load(strong_path)
            print(f"✓ Strong model (100 estimators): Loaded")

            self.model_weak = joblib.load(weak_path)
            print(f"✓ Weak model (50 estimators): Loaded")

            self.scaler = joblib.load(scaler_path)
            print(f"✓ Scaler: Loaded")

            with open(feature_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
            print(f"✓ Features ({len(self.feature_columns)}): {self.feature_columns}")

            print("\n" + "="*70)
            print("✅ ALL MODELS LOADED SUCCESSFULLY")
            print("="*70 + "\n")

        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            print("Run 2_train_rl_model.py first to train models")
            raise

    def load_available_stocks(self):
        """Load list of available stocks from dataset"""
        try:
            if os.path.exists(self.dataset_path):
                df = pd.read_csv(self.dataset_path)
                self.available_stocks = sorted(df['Stock'].unique().tolist())
                print(f"✓ Available stocks loaded: {len(self.available_stocks)} stocks")
        except Exception as e:
            print(f"Warning: Could not load stocks: {e}")

    def load_training_metrics(self):
        """Load training metrics for confidence assessment"""
        try:
            strong_metrics_path = os.path.join(self.model_dir, 'metrics_strong.pkl')
            weak_metrics_path = os.path.join(self.model_dir, 'metrics_weak.pkl')

            if os.path.exists(strong_metrics_path):
                with open(strong_metrics_path, 'rb') as f:
                    self.training_metrics['strong'] = pickle.load(f)

            if os.path.exists(weak_metrics_path):
                with open(weak_metrics_path, 'rb') as f:
                    self.training_metrics['weak'] = pickle.load(f)
        except Exception as e:
            print(f"Info: Training metrics not available: {e}")

    def get_stock_input(self):
        """Interactive stock selection with autocomplete"""
        print("\n" + "="*70)
        print("STOCK SELECTION")
        print("="*70)

        if self.available_stocks:
            print(f"\nAvailable stocks ({len(self.available_stocks)}):")
            print("  " + ", ".join(self.available_stocks[:15]))
            if len(self.available_stocks) > 15:
                print(f"  ... and {len(self.available_stocks) - 15} more")

            while True:
                stock_input = input("\nEnter stock name (case-insensitive, partial match ok): ").strip().upper()

                if not stock_input:
                    print("❌ Stock name cannot be empty!")
                    continue

                # Case-insensitive partial matching
                matches = [s for s in self.available_stocks if stock_input in s.upper()]

                if not matches:
                    print(f"❌ No stocks found matching '{stock_input}'")
                    print(f"   Try: {', '.join(self.available_stocks[:5])}")
                    continue

                if len(matches) == 1:
                    selected_stock = matches[0]
                    print(f"\n✓ Selected: {selected_stock}")
                    return selected_stock

                # Multiple matches
                print(f"\n🔍 Found {len(matches)} matches:")
                for i, stock in enumerate(matches[:10], 1):
                    print(f"   {i}. {stock}")

                try:
                    choice = int(input("\nEnter choice number (or 0 to search again): "))
                    if 1 <= choice <= len(matches):
                        selected_stock = matches[choice - 1]
                        print(f"\n✓ Selected: {selected_stock}")
                        return selected_stock
                    elif choice == 0:
                        continue
                    else:
                        print("❌ Invalid choice!")
                except ValueError:
                    print("❌ Please enter a valid number!")

        else:
            print("\n⚠️  No stocks loaded. Using generic mode.")
            stock_input = input("Enter stock name: ").strip()
            return stock_input if stock_input else "GENERIC"

    def get_parameters_input(self, stock_name):
        """Get 7 parameters with validation - CORRECTED"""
        print("\n" + "="*70)
        print(f"ENTER 7 PARAMETERS FOR {stock_name}")
        print("="*70)

        # CORRECTED: Only 7 features, matching training data
        param_names = [
            ("Price (0-1)", 0, 1),
            ("Volatility (0-1)", 0, 1),
            ("P/E Ratio (0-1)", 0, 1),
            ("Dividend (0-1)", 0, 1),
            ("RSI (0-1)", 0, 1),
            ("ATR (0-1)", 0, 1),
            ("Price Change (-1 to 1)", -1, 1)
        ]

        parameters = []

        for i, (name, min_val, max_val) in enumerate(param_names, 1):
            while True:
                try:
                    value = float(input(f"{i}. {name}: "))

                    if not (min_val <= value <= max_val):
                        print(f"   ❌ Must be between {min_val} and {max_val}!")
                        continue

                    parameters.append(value)
                    print(f"   ✓ Saved: {value:.4f}")
                    break

                except ValueError:
                    print("   ❌ Please enter a valid number!")

        return parameters

    def predict_from_parameters(self, stock_name, parameters):
        """Make prediction from 7 parameters - CORRECTED"""

        # CORRECTED: Check for 7 parameters
        if len(parameters) != 7:
            raise ValueError(f"Expected 7 parameters, got {len(parameters)}")

        X = np.array(parameters).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Predictions
        pred_strong = self.model_strong.predict(X_scaled)[0]
        prob_strong = self.model_strong.predict_proba(X_scaled)[0]

        pred_weak = self.model_weak.predict(X_scaled)[0]
        prob_weak = self.model_weak.predict_proba(X_scaled)[0]

        ensemble_prob = (prob_strong + prob_weak) / 2
        ensemble_pred = np.argmax(ensemble_prob)

        signal_map = {0: 'SELL', 1: 'BUY'}

        result = {
            'stock': stock_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_parameters': parameters,
            'signal_strong_100': signal_map[int(pred_strong)],
            'confidence_strong': max(prob_strong),
            'buy_prob_strong': prob_strong[1],
            'sell_prob_strong': prob_strong[0],

            'signal_weak_50': signal_map[int(pred_weak)],
            'confidence_weak': max(prob_weak),
            'buy_prob_weak': prob_weak[1],
            'sell_prob_weak': prob_weak[0],

            'ensemble_signal': signal_map[int(ensemble_pred)],
            'ensemble_confidence': max(ensemble_prob),
            'ensemble_buy_prob': ensemble_prob[1],
            'ensemble_sell_prob': ensemble_prob[0],
        }

        return result

    def calculate_reliability_score(self, result):
        """Calculate reliability score based on training metrics"""
        reliability = 0.0

        # Base score from confidence
        reliability += result['ensemble_confidence'] * 50

        # Agreement between models bonus
        if result['signal_strong_100'] == result['signal_weak_50']:
            reliability += 30
        else:
            reliability += 10

        # Training metrics factor
        if 'strong' in self.training_metrics:
            f1_score = self.training_metrics['strong'].get('F1-Score', 0.5)
            reliability += f1_score * 20

        return min(reliability / 100 * 100, 100)

    def get_recommendation(self, result):
        """Get detailed recommendation with confidence level"""
        reliability = self.calculate_reliability_score(result)

        if reliability >= 70:
            confidence_level = "🟢 HIGH"
        elif reliability >= 50:
            confidence_level = "🟡 MEDIUM"
        else:
            confidence_level = "🔴 LOW"

        return confidence_level, reliability

    def display_prediction(self, result):
        """Display detailed prediction with confidence analysis - CORRECTED"""
        stock_name = result['stock']

        print("\n" + "="*70)
        print(f"PREDICTION RESULT - {stock_name}")
        print("="*70)
        print(f"Time: {result['timestamp']}")

        # CORRECTED: Only 7 parameter names
        params_names = [
            "Price (0-1)",
            "Volatility (0-1)",
            "P/E Ratio (0-1)",
            "Dividend (0-1)",
            "RSI (0-1)",
            "ATR (0-1)",
            "Price Change (-1 to 1)"
        ]

        print("\nInput Parameters (7):")
        for i, (name, value) in enumerate(zip(params_names, result['input_parameters']), 1):
            print(f"  {i}. {name:28s} = {value:7.4f}")

        print("\n" + "-"*70)
        print("MODEL PREDICTIONS:")
        print("-"*70)

        # Strong model
        print(f"\n🔷 STRONG MODEL (100 Estimators):")
        print(f"   Signal:           {result['signal_strong_100']}")
        print(f"   Confidence:       {result['confidence_strong']:.2%}")
        print(f"   BUY Probability:  {result['buy_prob_strong']:.2%}")
        print(f"   SELL Probability: {result['sell_prob_strong']:.2%}")

        # Weak model
        print(f"\n🔶 WEAK MODEL (50 Estimators):")
        print(f"   Signal:           {result['signal_weak_50']}")
        print(f"   Confidence:       {result['confidence_weak']:.2%}")
        print(f"   BUY Probability:  {result['buy_prob_weak']:.2%}")
        print(f"   SELL Probability: {result['sell_prob_weak']:.2%}")

        # Ensemble
        print(f"\n🔵 ENSEMBLE (Average of Both):")
        print(f"   Signal:           {result['ensemble_signal']}")
        print(f"   Confidence:       {result['ensemble_confidence']:.2%}")
        print(f"   BUY Probability:  {result['ensemble_buy_prob']:.2%}")
        print(f"   SELL Probability: {result['ensemble_sell_prob']:.2%}")

        # Model agreement
        agreement = "✓ AGREE" if result['signal_strong_100'] == result['signal_weak_50'] else "✗ DISAGREE"
        print(f"\nModel Agreement: {agreement}")

        # Reliability analysis
        confidence_level, reliability_score = self.get_recommendation(result)

        print("\n" + "-"*70)
        print("RELIABILITY ANALYSIS:")
        print("-"*70)
        print(f"Confidence Level:   {confidence_level}")
        print(f"Reliability Score:  {reliability_score:.1f}/100")

        if reliability_score >= 70:
            risk_level = "Low Risk - Strong Signal"
        elif reliability_score >= 50:
            risk_level = "Medium Risk - Moderate Signal"
        else:
            risk_level = "High Risk - Weak Signal"

        print(f"Risk Level:         {risk_level}")

        print("\n" + "="*70)
        print(f"✅ FINAL RECOMMENDATION: {result['ensemble_signal']}")
        print("="*70)
        print(f"   Decision: {result['ensemble_signal'].upper()} {stock_name}")
        print(f"   Confidence: {confidence_level}")
        action = 'Consider buying' if result['ensemble_signal']=='BUY' else 'Consider selling'
        print(f"   Action: {action} with {risk_level.lower()}")
        print("="*70 + "\n")

    def save_prediction(self, result, filename=None):
        """Save single prediction to CSV - CORRECTED"""
        if filename is None:
            stock = result['stock'].replace('/', '_').replace('.', '_')
            filename = f"prediction_{stock}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # CORRECTED: Only 7 parameters
        csv_data = {
            'Stock': result['stock'],
            'Timestamp': result['timestamp'],
            'Price': result['input_parameters'][0],
            'Volatility': result['input_parameters'][1],
            'PE_Ratio': result['input_parameters'][2],
            'Dividend': result['input_parameters'][3],
            'RSI': result['input_parameters'][4],
            'ATR': result['input_parameters'][5],
            'Price_Change': result['input_parameters'][6],
            'Strong_Signal': result['signal_strong_100'],
            'Strong_Confidence': result['confidence_strong'],
            'Weak_Signal': result['signal_weak_50'],
            'Weak_Confidence': result['confidence_weak'],
            'Ensemble_Signal': result['ensemble_signal'],
            'Ensemble_Confidence': result['ensemble_confidence'],
        }

        df = pd.DataFrame([csv_data])
        df.to_csv(filename, index=False)
        print(f"✓ Prediction saved: {filename}\n")

    def interactive_prediction_mode(self):
        """Interactive mode for predictions"""
        while True:
            print("\n" + "="*70)
            print("INTERACTIVE PREDICTION MODE")
            print("="*70)
            print("1. Make prediction for a stock")
            print("2. Make multiple predictions (batch)")
            print("3. Exit")

            choice = input("\nEnter choice (1-3): ").strip()

            if choice == '1':
                stock = self.get_stock_input()
                params = self.get_parameters_input(stock)
                result = self.predict_from_parameters(stock, params)
                self.display_prediction(result)

                save_choice = input("Save prediction? (y/n): ").strip().lower()
                if save_choice == 'y':
                    self.save_prediction(result)

            elif choice == '2':
                print("\nBatch mode not yet implemented in interactive mode")

            elif choice == '3':
                print("\nGoodbye!")
                break

            else:
                print("❌ Invalid choice!")


if __name__ == "__main__":

    try:
        predictor = RLPredictionEngine(model_dir='./rl_models')

        # Start interactive mode
        predictor.interactive_prediction_mode()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
