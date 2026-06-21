SELECT AVG(close) AS avg_close_usd
FROM stock_prices
WHERE ticker = 'NVDA'
  AND CAST(date AS DATE) = DATE '2019-06-03';
