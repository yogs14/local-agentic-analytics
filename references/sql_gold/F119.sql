SELECT ticker, AVG(close) AS avg_close
FROM stock_prices
WHERE date BETWEEN DATE '2020-01-01' AND DATE '2020-01-31'
GROUP BY ticker
ORDER BY ticker;
