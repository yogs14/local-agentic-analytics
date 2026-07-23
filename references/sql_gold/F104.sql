SELECT MAX(open) AS value
FROM stock_prices
WHERE ticker = 'TSLA'
  AND date BETWEEN DATE '2019-01-01' AND DATE '2019-01-31';
