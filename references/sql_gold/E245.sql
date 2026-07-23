SELECT CAST(datetime AS DATE) AS day, AVG(Global_intensity) AS avg_value
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2007
  AND EXTRACT(MONTH FROM datetime) = 9
GROUP BY day
ORDER BY day;
