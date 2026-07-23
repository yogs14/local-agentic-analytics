SELECT MAX(volume) AS value
FROM stock_prices
WHERE ticker = 'TSLA'
  AND date BETWEEN DATE '2020-02-01' AND DATE '2020-02-29';
