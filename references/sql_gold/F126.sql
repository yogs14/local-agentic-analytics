SELECT AVG(high - low) AS avg_spread
FROM stock_prices
WHERE ticker = 'NFLX'
  AND date BETWEEN DATE '2019-02-01' AND DATE '2019-02-28';
