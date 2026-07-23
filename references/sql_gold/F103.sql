SELECT MAX(close) AS value
FROM stock_prices
WHERE ticker = 'GOOGL'
  AND date BETWEEN DATE '2019-03-01' AND DATE '2019-03-31';
