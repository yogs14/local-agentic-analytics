SELECT COUNT(*) AS day_count
FROM stock_prices
WHERE ticker = 'TSLA'
  AND close > 20.0;
