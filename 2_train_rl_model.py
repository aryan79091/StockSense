import pandas as pd
import numpy as np
import os
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, roc_auc_score, roc_curve, 
                             confusion_matrix, classification_report)

class RLStockTradingModel:
    """RL model for stock trading with 100 and 50 estimators"""

    def __init__(self, model_dir='./rl_models', dataset_path='./processed_data/rl_training_dataset.csv', metrics_dir='./metrics'):
        self.model_dir = model_dir
        self.dataset_path = dataset_path
        self.metrics_dir = metrics_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        self.model_strong = None  # 100 estimators
        self.model_weak = None    # 50 estimators
        self.scaler = StandardScaler()

        self.feature_columns = [
            'Price_Norm', 'Volatility_Norm', 'PE_Ratio_Norm',
            'Dividend_Norm', 'RSI_Norm', 'ATR_Norm', 'Price_Change_Norm'
        ]

        # Store metrics for later use
        self.metrics_strong = {}
        self.metrics_weak = {}

    def load_dataset(self):
        """Load prepared dataset"""
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset not found at: {self.dataset_path}")
            print("Run 1_dataset_preparation.py first")
            return None

        df = pd.read_csv(self.dataset_path)
        print(f"✓ Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")
        return df

    def prepare_features_and_labels(self, df):
        """Prepare features and generate labels"""
        df = df.sort_values(['Stock', 'Date']).reset_index(drop=True)

        labels = []

        for stock in df['Stock'].unique():
            stock_df = df[df['Stock'] == stock].copy()
            stock_prices = stock_df['Close'].values

            stock_labels = []
            for i in range(len(stock_prices) - 1):
                future_price = stock_prices[i + 1]
                current_price = stock_prices[i]

                if future_price > current_price:
                    stock_labels.append(1)  # BUY
                else:
                    stock_labels.append(0)  # SELL

            avg_label = np.mean(stock_labels) if stock_labels else 0.5
            stock_labels.append(1 if avg_label >= 0.5 else 0)
            labels.extend(stock_labels)

        df['Trading_Signal'] = labels

        available_features = [col for col in self.feature_columns if col in df.columns]

        X = df[available_features].fillna(0).values
        y = df['Trading_Signal'].values

        X_scaled = self.scaler.fit_transform(X)

        print(f"\n✓ Features shape: {X_scaled.shape}")
        print(f"✓ Labels shape: {y.shape}")
        buy_count = np.sum(y > 0.5)
        sell_count = len(y) - buy_count
        print(f"✓ BUY signals: {buy_count}, SELL signals: {sell_count}")

        return X_scaled, y, available_features

    def calculate_metrics(self, model, X, y, model_name="Model"):
        """Calculate comprehensive metrics"""
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

        metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred, zero_division=0),
            'Recall': recall_score(y, y_pred, zero_division=0),
            'F1-Score': f1_score(y, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y, y_pred_proba),
        }

        print(f"\n{'='*70}")
        print(f"{model_name} - PERFORMANCE METRICS")
        print(f"{'='*70}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name:15s}: {value:.4f}")

        return metrics, y_pred, y_pred_proba

    def plot_confusion_matrix(self, y_true, y_pred, model_name, filename):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['SELL', 'BUY'],
                    yticklabels=['SELL', 'BUY'])
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {filename}")

    def plot_roc_curve(self, y_true, y_pred_proba, model_name, filename):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{model_name} - ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {filename}")

    def plot_metrics_comparison(self):
        """Plot metrics comparison between models"""
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        strong_values = [self.metrics_strong[m] for m in metrics_names]
        weak_values = [self.metrics_weak[m] for m in metrics_names]

        x = np.arange(len(metrics_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, strong_values, width, label='Strong Model (100)', color='#2E86AB')
        bars2 = ax.bar(x + width/2, weak_values, width, label='Weak Model (50)', color='#A23B72')

        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison - Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: metrics_comparison.png")

    def plot_feature_importance(self):
        """Plot feature importance comparison"""
        importance_strong = self.model_strong.feature_importances_
        importance_weak = self.model_weak.feature_importances_

        x = np.arange(len(self.feature_columns))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, importance_strong, width, label='Strong Model (100)', color='#2E86AB')
        bars2 = ax.bar(x + width/2, importance_weak, width, label='Weak Model (50)', color='#A23B72')

        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.feature_columns, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: feature_importance.png")

    def save_metrics_report(self):
        """Save detailed metrics report as text"""
        report_file = os.path.join(self.metrics_dir, 'metrics_report.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("REINFORCEMENT LEARNING MODEL - PERFORMANCE METRICS\n")
            f.write("="*70 + "\n\n")

            f.write("STRONG MODEL (100 Estimators)\n")
            f.write("-"*70 + "\n")
            for metric_name, value in self.metrics_strong.items():
                f.write(f"  {metric_name:15s}: {value:.4f}\n")

            f.write("\n" + "="*70 + "\n\n")

            f.write("WEAK MODEL (50 Estimators)\n")
            f.write("-"*70 + "\n")
            for metric_name, value in self.metrics_weak.items():
                f.write(f"  {metric_name:15s}: {value:.4f}\n")

            f.write("\n" + "="*70 + "\n\n")

            f.write("COMPARISON\n")
            f.write("-"*70 + "\n")
            for metric_name in self.metrics_strong.keys():
                strong_val = self.metrics_strong[metric_name]
                weak_val = self.metrics_weak[metric_name]
                diff = strong_val - weak_val
                f.write(f"  {metric_name:15s}: Strong={strong_val:.4f}, Weak={weak_val:.4f}, Diff={diff:+.4f}\n")

            f.write("\n" + "="*70 + "\n")

        print(f"  ✓ Saved: {report_file}")

    def train_models(self, X, y):
        """Train both models"""
        print("\n" + "="*70)
        print("TRAINING REINFORCEMENT LEARNING MODELS")
        print("="*70)

        # Strong Model: 100 estimators
        print("\n[1/2] Training STRONG Model (100 estimators)...")
        self.model_strong = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        self.model_strong.fit(X, y)
        self.metrics_strong, y_pred_strong, y_pred_proba_strong = self.calculate_metrics(
            self.model_strong, X, y, "STRONG MODEL (100 Estimators)"
        )

        # Weak Model: 50 estimators
        print("\n[2/2] Training WEAK Model (50 estimators)...")
        self.model_weak = RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            min_samples_split=15,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        self.model_weak.fit(X, y)
        self.metrics_weak, y_pred_weak, y_pred_proba_weak = self.calculate_metrics(
            self.model_weak, X, y, "WEAK MODEL (50 Estimators)"
        )

        print("\n" + "="*70)

        return (self.metrics_strong, y_pred_strong, y_pred_proba_strong,
                self.metrics_weak, y_pred_weak, y_pred_proba_weak)

    def generate_visualizations(self, y, y_pred_strong, y_pred_proba_strong, 
                               y_pred_weak, y_pred_proba_weak):
        """Generate all visualization graphs"""
        print("\nGenerating visualizations...")
        print("-"*70)

        # Confusion matrices
        self.plot_confusion_matrix(y, y_pred_strong, "Strong Model (100)", 'confusion_matrix_strong.png')
        self.plot_confusion_matrix(y, y_pred_weak, "Weak Model (50)", 'confusion_matrix_weak.png')

        # ROC curves
        self.plot_roc_curve(y, y_pred_proba_strong, "Strong Model (100)", 'roc_curve_strong.png')
        self.plot_roc_curve(y, y_pred_proba_weak, "Weak Model (50)", 'roc_curve_weak.png')

        # Metrics comparison
        self.plot_metrics_comparison()

        # Feature importance
        self.plot_feature_importance()

    def save_models(self):
        """Save trained models and scaler"""
        print("\nSaving models...")
        print("-"*70)

        strong_path = os.path.join(self.model_dir, 'rl_model_strong_100.joblib')
        joblib.dump(self.model_strong, strong_path)
        print(f"✓ Strong model saved: {strong_path}")

        weak_path = os.path.join(self.model_dir, 'rl_model_weak_50.joblib')
        joblib.dump(self.model_weak, weak_path)
        print(f"✓ Weak model saved: {weak_path}")

        scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved: {scaler_path}")

        feature_path = os.path.join(self.model_dir, 'feature_columns.pkl')
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"✓ Feature columns saved: {feature_path}")

        # Save metrics
        metrics_strong_path = os.path.join(self.model_dir, 'metrics_strong.pkl')
        with open(metrics_strong_path, 'wb') as f:
            pickle.dump(self.metrics_strong, f)
        print(f"✓ Strong model metrics saved: {metrics_strong_path}")

        metrics_weak_path = os.path.join(self.model_dir, 'metrics_weak.pkl')
        with open(metrics_weak_path, 'wb') as f:
            pickle.dump(self.metrics_weak, f)
        print(f"✓ Weak model metrics saved: {metrics_weak_path}")

    def train_and_save(self):
        """Main training pipeline"""
        print("\n" + "="*70)
        print("REINFORCEMENT LEARNING MODEL TRAINING - COMPLETE PIPELINE")
        print("="*70)

        print("\nStep 1: Loading dataset...")
        df = self.load_dataset()

        if df is None:
            return False

        print("\nStep 2: Preparing features and labels...")
        X, y, available_features = self.prepare_features_and_labels(df)
        self.feature_columns = available_features

        print("\nStep 3: Training models...")
        results = self.train_models(X, y)
        y_pred_strong, y_pred_proba_strong = results[1], results[2]
        y_pred_weak, y_pred_proba_weak = results[4], results[5]

        print("\nStep 4: Generating visualizations...")
        self.generate_visualizations(y, y_pred_strong, y_pred_proba_strong, 
                                    y_pred_weak, y_pred_proba_weak)

        print("\nStep 5: Saving models and metrics...")
        self.save_models()

        print("\nStep 6: Generating metrics report...")
        self.save_metrics_report()

        print("\n" + "="*70)
        print("✅ MODEL TRAINING COMPLETE!")
        print("="*70)
        print(f"\nModels saved in: {self.model_dir}/")
        print(f"Metrics and graphs saved in: {self.metrics_dir}/")
        print("\nFiles created:")
        print("  Models:")
        print("    • rl_model_strong_100.joblib")
        print("    • rl_model_weak_50.joblib")
        print("    • scaler.joblib")
        print("    • feature_columns.pkl")
        print("    • metrics_strong.pkl")
        print("    • metrics_weak.pkl")
        print("\n  Metrics & Graphs:")
        print("    • metrics_report.txt (detailed text report)")
        print("    • confusion_matrix_strong.png")
        print("    • confusion_matrix_weak.png")
        print("    • roc_curve_strong.png")
        print("    • roc_curve_weak.png")
        print("    • metrics_comparison.png")
        print("    • feature_importance.png")
        print("\n" + "="*70 + "\n")

        return True


if __name__ == "__main__":
    trainer = RLStockTradingModel(
        model_dir='./rl_models',
        dataset_path='./processed_data/rl_training_dataset.csv',
        metrics_dir='./metrics'
    )
    trainer.train_and_save()
