SELECT SUM(volume) AS value
FROM stock_prices
WHERE ticker = 'TSLA'
  AND date BETWEEN DATE '2020-03-01' AND DATE '2020-03-31';
