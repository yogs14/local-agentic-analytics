SELECT EXTRACT(MONTH FROM datetime) AS month, AVG(Voltage) AS avg_value
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2008
GROUP BY month
HAVING AVG(Voltage) > 241.0
ORDER BY month;
