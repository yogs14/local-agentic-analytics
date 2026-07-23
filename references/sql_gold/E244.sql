SELECT CAST(datetime AS DATE) AS day, AVG(Voltage) AS avg_value
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2009
  AND EXTRACT(MONTH FROM datetime) = 3
GROUP BY day
ORDER BY day;
