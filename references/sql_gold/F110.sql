SELECT COUNT(*) AS trading_days
FROM stock_prices
WHERE ticker = 'GOOGL'
  AND date BETWEEN DATE '2019-09-01' AND DATE '2019-09-30';
