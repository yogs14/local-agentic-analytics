SELECT MAX(close) AS max_close_usd
FROM stock_prices
WHERE ticker = 'GOOGL'
  AND CAST(date AS DATE) BETWEEN DATE '2019-01-01' AND DATE '2019-03-31';
