SELECT ticker, AVG(close) AS avg_close
FROM stock_prices
WHERE date BETWEEN DATE '2019-08-01' AND DATE '2019-08-31'
GROUP BY ticker
ORDER BY ticker;
