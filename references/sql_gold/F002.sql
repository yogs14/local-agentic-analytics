SELECT MAX(close) AS max_close_usd
FROM stock_prices
WHERE ticker = 'TSLA';
