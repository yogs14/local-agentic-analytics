SELECT EXTRACT(MONTH FROM datetime) AS month, AVG(Global_intensity) AS avg_value
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2009
GROUP BY month
HAVING AVG(Global_intensity) > 4.0
ORDER BY month;
