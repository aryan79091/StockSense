def calculate_stock_metrics():
    print("\n📊 STOCK METRICS NORMALIZATION TOOL\n")

 
    current_price = float(input("Enter Current Price: "))
    open_price = float(input("Enter Open Price: "))
    high_price = float(input("Enter High Price: "))
    low_price = float(input("Enter Low Price: "))
    close_price = float(input("Enter Close Price: "))
    rsi = float(input("Enter RSI:"))
    atr = float(input("Enter ATR:"))
    pe_ratio = float(input("Enter P/E Ratio:"))
    dividend_yield = float(input("Enter Dividend Yield (%):"))


    price_norm = (current_price - low_price) / (high_price - low_price)
    volatility = high_price - low_price
    volatility_norm = volatility / low_price
    price_change = ((close_price - open_price) / open_price) * 100
    price_change_norm = (price_change + 10) / 20 if -10 <= price_change <= 10 else min(max(price_change / 100, 0), 1)
    rsi_norm = rsi / 100
    atr_norm = atr / current_price
    pe_ratio_norm = pe_ratio / 100
    dividend_norm = dividend_yield / 10


    print("\n📈 FINAL CALCULATED & NORMALIZED METRICS:\n")
    print(f"Price Change (%)       : {price_change:.2f}%")
    print(f"Volatility             : {volatility:.2f}")
    print(f"RSI                    : {rsi:.2f}")
    print(f"ATR                    : {atr:.2f}")
    print(f"P/E Ratio              : {pe_ratio:.2f}")
    print(f"Dividend Yield (%)     : {dividend_yield:.2f}%\n")


    print("🔹 Normalized Values (0 - 1 scale):")
    print(f"Price_Norm             : {price_norm:.3f}")
    print(f"Volatility_Norm        : {volatility_norm:.3f}")
    print(f"PE_Ratio_Norm          : {pe_ratio_norm:.3f}")
    print(f"Dividend_Norm          : {dividend_norm:.3f}")
    print(f"RSI_Norm               : {rsi_norm:.3f}")
    print(f"ATR_Norm               : {atr_norm:.3f}")
    print(f"Price_Change_Norm      : {price_change_norm:.3f}")


calculate_stock_metrics()
