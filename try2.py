import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
import joblib
import os
import pickle
from datetime import datetime

class AutocompleteCombobox(ttk.Combobox):
    """Combobox with autocomplete functionality"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._completion_list = []
        self._hits = []
        self._hit_index = 0
        self.position = 0
        
        self.bind('<KeyRelease>', self.handle_keyrelease)
        self['values'] = self._completion_list
    
    def set_completion_list(self, completion_list):
        """Set the completion list"""
        self._completion_list = sorted(completion_list, key=str.lower)
        self['values'] = self._completion_list
    
    def autocomplete(self, delta=0):
        """Autocomplete the combobox"""
        if delta:
            self.delete(self.position, tk.END)
        else:
            self.position = len(self.get())
        
        _hits = []
        for element in self._completion_list:
            if element.lower().startswith(self.get().lower()):
                _hits.append(element)
        
        if _hits != self._hits:
            self._hit_index = 0
            self._hits = _hits
        
        if _hits == self._hits and self._hits:
            self._hit_index = (self._hit_index + delta) % len(self._hits)
        
        if self._hits:
            self.delete(0, tk.END)
            self.insert(0, self._hits[self._hit_index])
            self.select_range(self.position, tk.END)
    
    def handle_keyrelease(self, event):
        """Handle key release event"""
        if event.keysym in ('BackSpace', 'Left', 'Right', 'Up', 'Down'):
            return
        
        if len(event.keysym) == 1:
            self.autocomplete()


class RLPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 AI Stock Prediction - Random Forest ML")
        self.root.geometry("1200x700")
        self.root.configure(bg="#131722")
        self.root.resizable(False, False)
        
        # Load models
        self.model_dir = './rl_models'
        self.dataset_path = './processed_data/rl_training_dataset.csv'
        self.model_strong = None
        self.model_weak = None
        self.scaler = None
        self.available_stocks = []
        
        self.load_models()
        self.load_available_stocks()
        
        self.create_widgets()
    
    def load_models(self):
        """Load trained models"""
        try:
            strong_path = os.path.join(self.model_dir, 'rl_model_strong_100.joblib')
            weak_path = os.path.join(self.model_dir, 'rl_model_weak_50.joblib')
            scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
            
            self.model_strong = joblib.load(strong_path)
            self.model_weak = joblib.load(weak_path)
            self.scaler = joblib.load(scaler_path)
            
        except FileNotFoundError as e:
            messagebox.showerror("Model Error", "Models not found! Run training script first.")
    
    def load_available_stocks(self):
        """Load available stocks from dataset"""
        try:
            if os.path.exists(self.dataset_path):
                df = pd.read_csv(self.dataset_path)
                self.available_stocks = sorted(df['Stock'].unique().tolist())
        except Exception as e:
            self.available_stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", 
                                     "BHARTIARTL", "SBIN", "HINDUNILVR", "ITC", "KOTAKBANK"]
    
    def create_widgets(self):
        """Create GUI widgets"""
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#131722")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title with AI icon
        title_frame = tk.Frame(main_frame, bg="#131722")
        title_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        title_label = tk.Label(
            title_frame,
            text="🤖 AI STOCK PREDICTION ENGINE - RANDOM FOREST ML",
            bg="#131722",
            fg="#ffffff",
            font=("Segoe UI", 16, "bold")
        )
        title_label.pack()
        
        subtitle = tk.Label(
            title_frame,
            text="Powered by Random Forest Algorithm | Ensemble Learning",
            bg="#131722",
            fg="#787b86",
            font=("Segoe UI", 9)
        )
        subtitle.pack()
        
        # Left Frame - Input Section
        input_frame = tk.Frame(main_frame, bg="#1e222d", relief=tk.RAISED, bd=1, 
                              width=450, highlightbackground="#2a2e39", highlightthickness=1)
        input_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        input_frame.grid_propagate(False)
        
        input_title = tk.Label(
            input_frame,
            text="🎯 INPUT PARAMETERS",
            bg="#1e222d",
            fg="#ffffff",
            font=("Segoe UI", 12, "bold")
        )
        input_title.pack(pady=15)
        
        # Stock Selection with Search
        stock_frame = tk.Frame(input_frame, bg="#1e222d")
        stock_frame.pack(padx=20, pady=10, fill=tk.X)
        
        tk.Label(
            stock_frame,
            text="Select Stock (Type to Search):",
            bg="#1e222d",
            fg="#d1d4dc",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))
        
        # Create autocomplete combobox
        self.stock_var = tk.StringVar()
        self.stock_combo = AutocompleteCombobox(
            stock_frame,
            textvariable=self.stock_var,
            font=("Consolas", 10),
            width=37
        )
        self.stock_combo.pack(fill=tk.X)
        self.stock_combo.set_completion_list(self.available_stocks)
        
        if self.available_stocks:
            self.stock_combo.set(self.available_stocks[0])
        
        # Style the combobox
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox',
                       fieldbackground='#2a2e39',
                       background='#2a2e39',
                       foreground='#ffffff',
                       arrowcolor='#ffffff',
                       bordercolor='#434651',
                       lightcolor='#2a2e39',
                       darkcolor='#2a2e39')
        style.map('TCombobox',
                 fieldbackground=[('readonly', '#2a2e39')],
                 selectbackground=[('readonly', '#434651')],
                 selectforeground=[('readonly', '#ffffff')])
        
        # Parameters Frame
        params_frame = tk.Frame(input_frame, bg="#1e222d")
        params_frame.pack(padx=20, pady=10)
        
        tk.Label(
            params_frame,
            text="Market Parameters (Normalized 0-1):",
            bg="#1e222d",
            fg="#d1d4dc",
            font=("Segoe UI", 10, "bold")
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Define parameter fields
        self.param_labels = [
            "Price (0-1):",
            "Volatility (0-1):",
            "P/E Ratio (0-1):",
            "Dividend (0-1):",
            "RSI (0-1):",
            "ATR (0-1):",
            "Price Change (-1 to 1):"
        ]
        
        self.param_entries = []
        
        for idx, label_text in enumerate(self.param_labels, 1):
            label = tk.Label(
                params_frame,
                text=label_text,
                bg="#1e222d",
                fg="#d1d4dc",
                font=("Segoe UI", 9),
                anchor="w",
                width=22
            )
            label.grid(row=idx, column=0, sticky="w", pady=6, padx=(0, 10))
            
            entry = tk.Entry(
                params_frame,
                bg="#2a2e39",
                fg="#ffffff",
                font=("Consolas", 10),
                width=18,
                insertbackground="#ffffff",
                relief=tk.FLAT,
                bd=2,
                highlightbackground="#434651",
                highlightthickness=1
            )
            entry.grid(row=idx, column=1, pady=6)
            self.param_entries.append(entry)
        
        # Buttons Frame
        button_frame = tk.Frame(input_frame, bg="#1e222d")
        button_frame.pack(pady=20, fill=tk.X, padx=20)
        
        button_container = tk.Frame(button_frame, bg="#1e222d")
        button_container.pack(anchor=tk.CENTER)
        
        predict_btn = tk.Button(
            button_container,
            text="🔮 PREDICT",
            command=self.make_prediction,
            bg="#363a45",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            width=15,
            height=1,
            relief=tk.FLAT,
            cursor="hand2",
            bd=0,
            activebackground="#4a4f5c",
            activeforeground="#ffffff"
        )
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(
            button_container,
            text="🗑️ CLEAR",
            command=self.clear_fields,
            bg="#363a45",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            width=15,
            height=1,
            relief=tk.FLAT,
            cursor="hand2",
            bd=0,
            activebackground="#4a4f5c",
            activeforeground="#ffffff"
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = tk.Button(
            button_container,
            text="💾 SAVE",
            command=self.save_prediction,
            bg="#363a45",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            width=15,
            height=1,
            relief=tk.FLAT,
            cursor="hand2",
            bd=0,
            activebackground="#4a4f5c",
            activeforeground="#ffffff"
        )
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Right Frame - Results
        result_frame = tk.Frame(main_frame, bg="#1e222d", relief=tk.RAISED, bd=1,
                               highlightbackground="#2a2e39", highlightthickness=1)
        result_frame.grid(row=1, column=1, sticky="nsew")
        
        result_title = tk.Label(
            result_frame,
            text="📊 PREDICTION RESULTS & ANALYSIS",
            bg="#1e222d",
            fg="#ffffff",
            font=("Segoe UI", 12, "bold")
        )
        result_title.pack(pady=15)
        
        # Result text area
        text_frame = tk.Frame(result_frame, bg="#1e222d")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        scrollbar = tk.Scrollbar(text_frame, bg="#2a2e39", troughcolor="#1e222d")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(
            text_frame,
            bg="#2a2e39",
            fg="#d1d4dc",
            font=("Consolas", 10),
            wrap=tk.WORD,
            relief=tk.FLAT,
            bd=0,
            yscrollcommand=scrollbar.set,
            state='disabled',
            insertbackground="#ffffff"
        )
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.result_text.yview)
        
        # Text tags for styling - COLORFUL VERSION
        self.result_text.tag_configure("header", foreground="#ffffff", font=("Consolas", 11, "bold"))
        self.result_text.tag_configure("header_green", foreground="#26a69a", font=("Consolas", 11, "bold"))
        self.result_text.tag_configure("subheader", foreground="#b2b5be", font=("Consolas", 10, "bold"))
        self.result_text.tag_configure("subheader_green", foreground="#26a69a", font=("Consolas", 10, "bold"))
        self.result_text.tag_configure("subheader_blue", foreground="#42a5f5", font=("Consolas", 10, "bold"))
        self.result_text.tag_configure("subheader_orange", foreground="#ffa726", font=("Consolas", 10, "bold"))
        self.result_text.tag_configure("separator", foreground="#434651")
        self.result_text.tag_configure("label", foreground="#787b86")
        self.result_text.tag_configure("value", foreground="#d1d4dc", font=("Consolas", 10, "bold"))
        
        # Green colors for BUY signals
        self.result_text.tag_configure("buy_green", foreground="#26a69a", font=("Consolas", 11, "bold"))
        self.result_text.tag_configure("buy_green_large", foreground="#26a69a", font=("Consolas", 12, "bold"))
        self.result_text.tag_configure("green_text", foreground="#4caf50", font=("Consolas", 10, "bold"))
        
        # Red colors for SELL signals
        self.result_text.tag_configure("sell_red", foreground="#ef5350", font=("Consolas", 11, "bold"))
        self.result_text.tag_configure("sell_red_large", foreground="#ef5350", font=("Consolas", 12, "bold"))
        self.result_text.tag_configure("red_text", foreground="#f44336", font=("Consolas", 10, "bold"))
        
        # Yellow colors for warnings/medium confidence
        self.result_text.tag_configure("yellow_text", foreground="#ffeb3b", font=("Consolas", 10, "bold"))
        self.result_text.tag_configure("orange_text", foreground="#ff9800", font=("Consolas", 10, "bold"))
        
        self.result_text.tag_configure("high", foreground="#ffffff", font=("Consolas", 10, "bold"))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=0, minsize=450)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        self.current_result = None
    
    def validate_inputs(self):
        """Validate all parameter inputs"""
        try:
            parameters = []
            for i, entry in enumerate(self.param_entries):
                value = float(entry.get())
                
                # Check range based on parameter
                if i < 6:  # First 6 parameters (0-1)
                    if not (0 <= value <= 1):
                        raise ValueError(f"Parameter {i+1} must be between 0 and 1")
                else:  # Price Change (-1 to 1)
                    if not (-1 <= value <= 1):
                        raise ValueError(f"Price Change must be between -1 and 1")
                
                parameters.append(value)
            
            return parameters
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return None
    
    def make_prediction(self):
        """Make prediction using trained models"""
        stock_name = self.stock_var.get().strip()
        if not stock_name:
            messagebox.showwarning("Warning", "Please select or enter a stock name!")
            return
        
        parameters = self.validate_inputs()
        if parameters is None:
            return
        
        if self.model_strong is None or self.model_weak is None:
            messagebox.showerror("Error", "Models not loaded!")
            return
        
        try:
            # Prepare data
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
                'parameters': parameters,
                'signal_strong': signal_map[int(pred_strong)],
                'confidence_strong': max(prob_strong),
                'buy_prob_strong': prob_strong[1],
                'sell_prob_strong': prob_strong[0],
                'signal_weak': signal_map[int(pred_weak)],
                'confidence_weak': max(prob_weak),
                'buy_prob_weak': prob_weak[1],
                'sell_prob_weak': prob_weak[0],
                'ensemble_signal': signal_map[int(ensemble_pred)],
                'ensemble_confidence': max(ensemble_prob),
                'ensemble_buy_prob': ensemble_prob[1],
                'ensemble_sell_prob': ensemble_prob[0],
            }
            
            self.current_result = result
            self.display_prediction(result)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
    
    def display_prediction(self, result):
        """Display prediction results with colorful indicators"""
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        
        # Header
        self.result_text.insert(tk.END, "═" * 70 + "\n", "separator")
        self.result_text.insert(tk.END, f"  PREDICTION FOR: {result['stock']}\n", "header")
        self.result_text.insert(tk.END, "═" * 70 + "\n", "separator")
        self.result_text.insert(tk.END, f"  Time: {result['timestamp']}\n\n", "label")
        
        # Input Parameters
        self.result_text.insert(tk.END, "📝 INPUT PARAMETERS:\n", "subheader")
        self.result_text.insert(tk.END, "─" * 70 + "\n", "separator")
        
        param_names = ["Price", "Volatility", "P/E Ratio", "Dividend", "RSI", "ATR", "Price Change"]
        for i, (name, value) in enumerate(zip(param_names, result['parameters']), 1):
            self.result_text.insert(tk.END, f"  {i}. {name:15s} = ", "label")
            self.result_text.insert(tk.END, f"{value:7.4f}\n", "value")
        
        self.result_text.insert(tk.END, "\n")
        
        # Model Predictions
        self.result_text.insert(tk.END, "🤖 RANDOM FOREST MODEL PREDICTIONS:\n", "subheader_green")
        self.result_text.insert(tk.END, "═" * 70 + "\n", "separator")
        
        # Strong Model
        self.result_text.insert(tk.END, "\n🔷 STRONG MODEL (100 Trees):\n", "subheader_blue")
        self.result_text.insert(tk.END, f"  Signal:           ", "label")
        tag = "buy_green" if result['signal_strong'] == 'BUY' else "sell_red"
        self.result_text.insert(tk.END, f"{result['signal_strong']}\n", tag)
        self.result_text.insert(tk.END, f"  Confidence:       ", "label")
        self.result_text.insert(tk.END, f"{result['confidence_strong']:.2%}\n", "value")
        self.result_text.insert(tk.END, f"  BUY Probability:  ", "label")
        self.result_text.insert(tk.END, f"{result['buy_prob_strong']:.2%}\n", "green_text")
        self.result_text.insert(tk.END, f"  SELL Probability: ", "label")
        self.result_text.insert(tk.END, f"{result['sell_prob_strong']:.2%}\n", "red_text")
        
        # Weak Model
        self.result_text.insert(tk.END, "\n🔶 WEAK MODEL (50 Trees):\n", "subheader_orange")
        self.result_text.insert(tk.END, f"  Signal:           ", "label")
        tag = "buy_green" if result['signal_weak'] == 'BUY' else "sell_red"
        self.result_text.insert(tk.END, f"{result['signal_weak']}\n", tag)
        self.result_text.insert(tk.END, f"  Confidence:       ", "label")
        self.result_text.insert(tk.END, f"{result['confidence_weak']:.2%}\n", "value")
        self.result_text.insert(tk.END, f"  BUY Probability:  ", "label")
        self.result_text.insert(tk.END, f"{result['buy_prob_weak']:.2%}\n", "green_text")
        self.result_text.insert(tk.END, f"  SELL Probability: ", "label")
        self.result_text.insert(tk.END, f"{result['sell_prob_weak']:.2%}\n", "red_text")
        
        # Ensemble
        self.result_text.insert(tk.END, "\n🔵 ENSEMBLE PREDICTION (Average):\n", "subheader_blue")
        self.result_text.insert(tk.END, f"  Signal:           ", "label")
        tag = "buy_green" if result['ensemble_signal'] == 'BUY' else "sell_red"
        self.result_text.insert(tk.END, f"{result['ensemble_signal']}\n", tag)
        self.result_text.insert(tk.END, f"  Confidence:       ", "label")
        self.result_text.insert(tk.END, f"{result['ensemble_confidence']:.2%}\n", "value")
        self.result_text.insert(tk.END, f"  BUY Probability:  ", "label")
        self.result_text.insert(tk.END, f"{result['ensemble_buy_prob']:.2%}\n", "green_text")
        self.result_text.insert(tk.END, f"  SELL Probability: ", "label")
        self.result_text.insert(tk.END, f"{result['ensemble_sell_prob']:.2%}\n", "red_text")
        
        # Agreement
        if result['signal_strong'] == result['signal_weak']:
            agreement = "✓ MODELS AGREE"
            agreement_tag = "green_text"
        else:
            agreement = "✗ MODELS DISAGREE"
            agreement_tag = "yellow_text"
        
        self.result_text.insert(tk.END, f"\n  Model Agreement: ", "label")
        self.result_text.insert(tk.END, f"{agreement}\n", agreement_tag)
        
        # Final Recommendation
        reliability = result['ensemble_confidence'] * 100
        if reliability >= 70:
            confidence_level = "🟢 HIGH CONFIDENCE"
            confidence_tag = "green_text"
            risk = "Low Risk"
            risk_tag = "green_text"
        elif reliability >= 50:
            confidence_level = "🟡 MEDIUM CONFIDENCE"
            confidence_tag = "yellow_text"
            risk = "Medium Risk"
            risk_tag = "yellow_text"
        else:
            confidence_level = "🔴 LOW CONFIDENCE"
            confidence_tag = "red_text"
            risk = "High Risk"
            risk_tag = "red_text"
        
        self.result_text.insert(tk.END, "\n")
        self.result_text.insert(tk.END, "═" * 70 + "\n", "separator")
        self.result_text.insert(tk.END, "  ✅ FINAL RECOMMENDATION\n", "header_green")
        self.result_text.insert(tk.END, "═" * 70 + "\n", "separator")
        self.result_text.insert(tk.END, f"  Signal:      ", "label")
        tag = "buy_green_large" if result['ensemble_signal'] == 'BUY' else "sell_red_large"
        self.result_text.insert(tk.END, f"{result['ensemble_signal']} {result['stock']}\n", tag)
        self.result_text.insert(tk.END, f"  Confidence:  ", "label")
        self.result_text.insert(tk.END, f"{confidence_level}\n", confidence_tag)
        self.result_text.insert(tk.END, f"  Reliability: ", "label")
        self.result_text.insert(tk.END, f"{reliability:.1f}/100\n", "value")
        self.result_text.insert(tk.END, f"  Risk Level:  ", "label")
        self.result_text.insert(tk.END, f"{risk}\n", risk_tag)
        self.result_text.insert(tk.END, "═" * 70 + "\n", "separator")
        
        self.result_text.config(state='disabled')
    
    def clear_fields(self):
        """Clear all input fields"""
        for entry in self.param_entries:
            entry.delete(0, tk.END)
        
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state='disabled')
        
        self.current_result = None
    
    def save_prediction(self):
        """Save prediction to CSV"""
        if self.current_result is None:
            messagebox.showwarning("Warning", "No prediction to save!")
            return
        
        try:
            result = self.current_result
            stock = result['stock'].replace('/', '_').replace('.', '_')
            filename = f"prediction_{stock}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            csv_data = {
                'Stock': result['stock'],
                'Timestamp': result['timestamp'],
                'Price': result['parameters'][0],
                'Volatility': result['parameters'][1],
                'PE_Ratio': result['parameters'][2],
                'Dividend': result['parameters'][3],
                'RSI': result['parameters'][4],
                'ATR': result['parameters'][5],
                'Price_Change': result['parameters'][6],
                'Strong_Signal': result['signal_strong'],
                'Strong_Confidence': result['confidence_strong'],
                'Weak_Signal': result['signal_weak'],
                'Weak_Confidence': result['confidence_weak'],
                'Ensemble_Signal': result['ensemble_signal'],
                'Ensemble_Confidence': result['ensemble_confidence'],
            }
            
            df = pd.DataFrame([csv_data])
            df.to_csv(filename, index=False)
            
            messagebox.showinfo("Success", f"Prediction saved to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Save Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = RLPredictionGUI(root)
    root.mainloop()
