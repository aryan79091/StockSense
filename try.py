import tkinter as tk
from tkinter import ttk, messagebox

def calculate_metrics():
    try:
        # Get input values
        current_price = float(entry_current.get())
        open_price = float(entry_open.get())
        high_price = float(entry_high.get())
        low_price = float(entry_low.get())
        close_price = float(entry_close.get())
        rsi = float(entry_rsi.get())
        atr = float(entry_atr.get())
        pe_ratio = float(entry_pe.get())
        dividend_yield = float(entry_dividend.get())
        
        # Calculate metrics
        price_norm = (current_price - low_price) / (high_price - low_price)
        volatility = high_price - low_price
        volatility_norm = volatility / low_price
        price_change = ((close_price - open_price) / open_price) * 100
        price_change_norm = (price_change + 10) / 20 if -10 <= price_change <= 10 else min(max(price_change / 100, 0), 1)
        rsi_norm = rsi / 100
        atr_norm = atr / current_price
        pe_ratio_norm = pe_ratio / 100
        dividend_norm = dividend_yield / 10
        
        # Clear previous results
        result_text.config(state='normal')
        result_text.delete(1.0, tk.END)
        
        # Display calculated metrics
        result_text.insert(tk.END, "📈 CALCULATED METRICS\n", "header")
        result_text.insert(tk.END, "─" * 50 + "\n\n", "separator")
        
        result_text.insert(tk.END, f"Price Change:          ", "label")
        result_text.insert(tk.END, f"{price_change:.2f}%\n", "positive" if price_change >= 0 else "negative")
        
        result_text.insert(tk.END, f"Volatility:            ", "label")
        result_text.insert(tk.END, f"{volatility:.2f}\n", "value")
        
        result_text.insert(tk.END, f"RSI:                   ", "label")
        result_text.insert(tk.END, f"{rsi:.2f}\n", "value")
        
        result_text.insert(tk.END, f"ATR:                   ", "label")
        result_text.insert(tk.END, f"{atr:.2f}\n", "value")
        
        result_text.insert(tk.END, f"P/E Ratio:             ", "label")
        result_text.insert(tk.END, f"{pe_ratio:.2f}\n", "value")
        
        result_text.insert(tk.END, f"Dividend Yield:        ", "label")
        result_text.insert(tk.END, f"{dividend_yield:.2f}%\n\n", "value")
        
        # Display normalized metrics
        result_text.insert(tk.END, "\n🔹 NORMALIZED VALUES (0-1 Scale)\n", "header")
        result_text.insert(tk.END, "─" * 50 + "\n\n", "separator")
        
        result_text.insert(tk.END, f"Price Norm:            ", "label")
        result_text.insert(tk.END, f"{price_norm:.3f}\n", "norm")
        
        result_text.insert(tk.END, f"Volatility Norm:       ", "label")
        result_text.insert(tk.END, f"{volatility_norm:.3f}\n", "norm")
        
        result_text.insert(tk.END, f"PE Ratio Norm:         ", "label")
        result_text.insert(tk.END, f"{pe_ratio_norm:.3f}\n", "norm")
        
        result_text.insert(tk.END, f"Dividend Norm:         ", "label")
        result_text.insert(tk.END, f"{dividend_norm:.3f}\n", "norm")
        
        result_text.insert(tk.END, f"RSI Norm:              ", "label")
        result_text.insert(tk.END, f"{rsi_norm:.3f}\n", "norm")
        
        result_text.insert(tk.END, f"ATR Norm:              ", "label")
        result_text.insert(tk.END, f"{atr_norm:.3f}\n", "norm")
        
        result_text.insert(tk.END, f"Price Change Norm:     ", "label")
        result_text.insert(tk.END, f"{price_change_norm:.3f}\n", "norm")
        
        result_text.config(state='disabled')
        
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields!")

def clear_fields():
    entry_current.delete(0, tk.END)
    entry_open.delete(0, tk.END)
    entry_high.delete(0, tk.END)
    entry_low.delete(0, tk.END)
    entry_close.delete(0, tk.END)
    entry_rsi.delete(0, tk.END)
    entry_atr.delete(0, tk.END)
    entry_pe.delete(0, tk.END)
    entry_dividend.delete(0, tk.END)
    result_text.config(state='normal')
    result_text.delete(1.0, tk.END)
    result_text.config(state='disabled')

# Create main window
root = tk.Tk()
root.title("📊 Stock Metrics Normalization Tool")
root.geometry("1100x650")
root.configure(bg="#131722")  # Dark background like TradingView
root.resizable(False, False)

# Style configuration
style = ttk.Style()
style.theme_use('clam')
style.configure('TLabel', background="#131722", foreground="#d1d4dc", font=("Segoe UI", 10))
style.configure('Header.TLabel', background="#131722", foreground="#ffffff", font=("Segoe UI", 14, "bold"))

# Main container
main_frame = tk.Frame(root, bg="#131722")
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Title
title_label = ttk.Label(main_frame, text="📊 STOCK METRICS CALCULATOR", style='Header.TLabel')
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

# Left Frame - Input Fields
input_frame = tk.Frame(main_frame, bg="#1e222d", relief=tk.RAISED, bd=1, width=400, highlightbackground="#2a2e39", highlightthickness=1)
input_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
input_frame.grid_propagate(False)

input_title = tk.Label(input_frame, text="📝 INPUT PARAMETERS", bg="#1e222d", fg="#ffffff", font=("Segoe UI", 12, "bold"))
input_title.pack(pady=15)

# Input fields container
fields_frame = tk.Frame(input_frame, bg="#1e222d")
fields_frame.pack(padx=20, pady=10)

# Define input fields
fields = [
    ("Current Price:", "entry_current"),
    ("Open Price:", "entry_open"),
    ("High Price:", "entry_high"),
    ("Low Price:", "entry_low"),
    ("Close Price:", "entry_close"),
    ("RSI:", "entry_rsi"),
    ("ATR:", "entry_atr"),
    ("P/E Ratio:", "entry_pe"),
    ("Dividend Yield (%):", "entry_dividend")
]

entries = {}
for idx, (label_text, entry_name) in enumerate(fields):
    label = tk.Label(fields_frame, text=label_text, bg="#1e222d", fg="#d1d4dc", font=("Segoe UI", 10), anchor="w", width=18)
    label.grid(row=idx, column=0, sticky="w", pady=8, padx=(0, 10))
    
    entry = tk.Entry(fields_frame, bg="#2a2e39", fg="#ffffff", font=("Consolas", 10), width=18, insertbackground="#ffffff", relief=tk.FLAT, bd=2, highlightbackground="#434651", highlightthickness=1)
    entry.grid(row=idx, column=1, pady=8)
    entries[entry_name] = entry

# Assign entries to variables
entry_current = entries["entry_current"]
entry_open = entries["entry_open"]
entry_high = entries["entry_high"]
entry_low = entries["entry_low"]
entry_close = entries["entry_close"]
entry_rsi = entries["entry_rsi"]
entry_atr = entries["entry_atr"]
entry_pe = entries["entry_pe"]
entry_dividend = entries["entry_dividend"]

# Buttons Frame
button_frame = tk.Frame(input_frame, bg="#1e222d")
button_frame.pack(pady=20, fill=tk.X, padx=20)

button_container = tk.Frame(button_frame, bg="#1e222d")
button_container.pack(anchor=tk.CENTER)

calculate_btn = tk.Button(button_container, text="🔍 CALCULATE", command=calculate_metrics, bg="#363a45", fg="#ffffff", font=("Segoe UI", 10, "bold"), width=15, height=1, relief=tk.FLAT, cursor="hand2", bd=0, activebackground="#4a4f5c", activeforeground="#ffffff")
calculate_btn.pack(side=tk.LEFT, padx=5)

clear_btn = tk.Button(button_container, text="🗑️ CLEAR", command=clear_fields, bg="#363a45", fg="#ffffff", font=("Segoe UI", 10, "bold"), width=15, height=1, relief=tk.FLAT, cursor="hand2", bd=0, activebackground="#4a4f5c", activeforeground="#ffffff")
clear_btn.pack(side=tk.LEFT, padx=5)

# Right Frame - Results
result_frame = tk.Frame(main_frame, bg="#1e222d", relief=tk.RAISED, bd=1, highlightbackground="#2a2e39", highlightthickness=1)
result_frame.grid(row=1, column=1, sticky="nsew")

result_title = tk.Label(result_frame, text="📊 RESULTS", bg="#1e222d", fg="#ffffff", font=("Segoe UI", 12, "bold"))
result_title.pack(pady=15)

# Result text widget with scrollbar
text_frame = tk.Frame(result_frame, bg="#1e222d")
text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

scrollbar = tk.Scrollbar(text_frame, bg="#2a2e39", troughcolor="#1e222d", activebackground="#434651")
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

result_text = tk.Text(text_frame, bg="#2a2e39", fg="#d1d4dc", font=("Consolas", 11), wrap=tk.WORD, relief=tk.FLAT, bd=0, yscrollcommand=scrollbar.set, state='disabled', insertbackground="#ffffff")
result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar.config(command=result_text.yview)

# Text tags for styling - Black & White theme
result_text.tag_configure("header", foreground="#ffffff", font=("Consolas", 12, "bold"))
result_text.tag_configure("separator", foreground="#434651")
result_text.tag_configure("label", foreground="#787b86")
result_text.tag_configure("value", foreground="#d1d4dc", font=("Consolas", 11, "bold"))
result_text.tag_configure("positive", foreground="#b2b5be", font=("Consolas", 11, "bold"))
result_text.tag_configure("negative", foreground="#787b86", font=("Consolas", 11, "bold"))
result_text.tag_configure("norm", foreground="#d1d4dc", font=("Consolas", 11, "bold"))

# Configure grid weights
main_frame.columnconfigure(0, weight=0, minsize=400)
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(1, weight=1)

# Run the application
root.mainloop()
