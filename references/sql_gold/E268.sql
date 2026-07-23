SELECT EXTRACT(ISODOW FROM datetime) AS weekday, AVG(Voltage) AS avg_value
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2008
GROUP BY weekday
ORDER BY weekday;
