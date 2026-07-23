SELECT AVG(high - low) AS avg_spread
FROM stock_prices
WHERE ticker = 'TSLA'
  AND date BETWEEN DATE '2019-12-01' AND DATE '2019-12-31';
