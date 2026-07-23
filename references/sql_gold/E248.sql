SELECT EXTRACT(HOUR FROM datetime) AS hour, AVG(Global_active_power) AS avg_kw
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2007
  AND EXTRACT(MONTH FROM datetime) = 12
GROUP BY hour
ORDER BY avg_kw DESC, hour ASC
LIMIT 3;
