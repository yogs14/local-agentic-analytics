SELECT date, volume
FROM stock_prices
WHERE ticker = 'NFLX'
ORDER BY volume DESC, date ASC
LIMIT 5;
