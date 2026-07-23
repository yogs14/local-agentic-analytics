SELECT MIN(open) AS value
FROM stock_prices
WHERE ticker = 'TSLA'
  AND date BETWEEN DATE '2019-10-01' AND DATE '2019-10-31';
