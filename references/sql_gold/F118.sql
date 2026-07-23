SELECT date, close
FROM stock_prices
WHERE ticker = 'TSLA'
ORDER BY close ASC, date ASC
LIMIT 1;
