SELECT EXTRACT(MONTH FROM datetime) AS month, AVG(Global_active_power) AS avg_value
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2007
GROUP BY month
HAVING AVG(Global_active_power) > 1.0
ORDER BY month;
