param(
  [string[]]$Tickers = @("VCB","CII","MSN"),
  [string]$DataDir = ".",
  [string]$PriceVol = "Price_Vol.xlsx",
  [ValidateSet("FLAT","HOLDING")] [string]$PositionMode = "FLAT",
  [ValidateSet("D","W")] [string]$Timeframe = "D",
  [switch]$Strict
)

$strictArg = ""
if ($Strict) { $strictArg = "--strict" }

python tools/check_golden_narrative.py --tickers $Tickers --data-dir $DataDir --price-vol $PriceVol --position-mode $PositionMode --timeframe $Timeframe $strictArg
exit $LASTEXITCODE
