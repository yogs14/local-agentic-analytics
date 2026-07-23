SELECT MIN(close) AS value
FROM stock_prices
WHERE ticker = 'TSLA'
  AND date BETWEEN DATE '2019-07-01' AND DATE '2019-07-31';
