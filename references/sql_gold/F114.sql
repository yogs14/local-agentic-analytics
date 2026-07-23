SELECT EXTRACT(MONTH FROM date) AS month, AVG(close) AS avg_close
FROM stock_prices
WHERE ticker = 'GOOGL'
  AND EXTRACT(YEAR FROM date) = 2019
GROUP BY month
ORDER BY month;
