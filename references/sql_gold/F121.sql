SELECT EXTRACT(MONTH FROM date) AS month, AVG(close) AS avg_close
FROM stock_prices
WHERE ticker = 'NVDA'
  AND EXTRACT(YEAR FROM date) = 2019
GROUP BY month
HAVING AVG(close) > 5.0
ORDER BY month;
