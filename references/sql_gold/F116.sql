SELECT date, volume
FROM stock_prices
WHERE ticker = 'NVDA'
ORDER BY volume DESC, date ASC
LIMIT 3;
