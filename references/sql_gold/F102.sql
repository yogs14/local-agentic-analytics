SELECT AVG(open) AS value
FROM stock_prices
WHERE ticker = 'GOOGL'
  AND date BETWEEN DATE '2019-11-01' AND DATE '2019-11-30';
