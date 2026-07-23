SELECT AVG(volume) AS value
FROM stock_prices
WHERE ticker = 'GOOGL'
  AND date BETWEEN DATE '2019-05-01' AND DATE '2019-05-31';
