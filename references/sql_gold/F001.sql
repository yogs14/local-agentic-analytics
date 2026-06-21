SELECT AVG(close) AS avg_close_usd
FROM stock_prices
WHERE ticker = 'NVDA'
  AND CAST(date AS DATE) BETWEEN DATE '2019-01-02' AND DATE '2019-01-31';
