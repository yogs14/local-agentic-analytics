SELECT COUNT(*) AS trading_days
FROM stock_prices
WHERE ticker = 'NVDA'
  AND date >= DATE '2019-01-01' AND date < DATE '2020-01-01';
