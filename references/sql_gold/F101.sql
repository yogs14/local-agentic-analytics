SELECT AVG(close) AS value
FROM stock_prices
WHERE ticker = 'TSLA'
  AND date BETWEEN DATE '2020-04-01' AND DATE '2020-04-30';
