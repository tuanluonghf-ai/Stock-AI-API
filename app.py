from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 1. Define the input: GPT sends these calculated numbers
class MarketIndicators(BaseModel):
    ticker: str
    price: float
    rsi: float
    macd_signal: str  # "Bullish" or "Bearish"
    ma_trend: str     # "Above MA20" or "Below MA20"
    vol_ratio: float  # Current Vol / Avg20 Vol (e.g., 1.2 for 120%)

@app.post("/strategy_engine")
def run_strategy(data: MarketIndicators):
    # --- YOUR SECRET MASTER LOGIC STARTS HERE ---
    
    # Logic A: 12-Scenario Classification (Simplified Example)
    scenario = "Neutral Consolidation"
    if data.price > 0: # Placeholder check
        if data.ma_trend == "Above MA20" and data.vol_ratio > 1.1:
            scenario = "Strong Breakout (Scenario #1)"
        elif data.rsi < 30:
            scenario = "Oversold Bounce (Scenario #9)"
        elif data.rsi > 70:
            scenario = "Overbought Correction (Scenario #4)"

    # Logic B: Calculate Conviction Score (0-100)
    score = 50
    if "Breakout" in scenario: score += 30
    if data.macd_signal == "Bullish": score += 10
    if data.vol_ratio > 1.5: score += 10
    
    # Logic C: R:R Simulation (Dynamic Calculation)
    stop_loss = round(data.price * 0.96, 2) # Example: 4% risk
    take_profit = round(data.price * 1.08, 2) # Example: 8% reward
    rr_ratio = round((take_profit - data.price) / (data.price - stop_loss), 2)

    # --- OUTPUT TO GPT ---
    return {
        "scenario_classification": scenario,
        "conviction_score": score,
        "trade_plan": {
            "action": "BUY" if score > 70 else "WATCH",
            "entry_zone": f"{data.price} - {round(data.price*1.01, 2)}",
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "rr_ratio": rr_ratio
        },
        "narrative": f"Based on Volume of {int(data.vol_ratio*100)}% avg, the stock is in a {scenario} phase."
    }