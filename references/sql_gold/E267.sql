SELECT EXTRACT(ISODOW FROM datetime) AS weekday, AVG(Global_active_power) AS avg_value
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2009
GROUP BY weekday
ORDER BY weekday;
