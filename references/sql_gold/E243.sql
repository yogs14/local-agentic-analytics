SELECT CAST(datetime AS DATE) AS day, AVG(Global_active_power) AS avg_value
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2009
  AND EXTRACT(MONTH FROM datetime) = 8
GROUP BY day
ORDER BY day;
