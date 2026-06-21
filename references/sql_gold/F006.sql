SELECT AVG(close) AS avg_close_usd
FROM stock_prices
WHERE ticker = 'TSLA'
  AND CAST(date AS DATE) BETWEEN DATE '2020-01-01' AND DATE '2020-06-10';
