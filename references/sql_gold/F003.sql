SELECT MIN(close) AS min_close_usd
FROM stock_prices
WHERE ticker = 'GOOGL';
