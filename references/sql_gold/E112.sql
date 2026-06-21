SELECT
    STRFTIME(datetime, '%Y-%m') AS month,
    AVG(Voltage) AS avg_voltage_v
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2007
GROUP BY month
ORDER BY month;
