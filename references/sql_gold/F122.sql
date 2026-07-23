SELECT EXTRACT(MONTH FROM date) AS month, AVG(close) AS avg_close
FROM stock_prices
WHERE ticker = 'TSLA'
  AND EXTRACT(YEAR FROM date) = 2019
GROUP BY month
HAVING AVG(close) > 20.0
ORDER BY month;
