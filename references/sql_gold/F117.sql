SELECT date, close
FROM stock_prices
WHERE ticker = 'NVDA'
ORDER BY close DESC, date ASC
LIMIT 1;
