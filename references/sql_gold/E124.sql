SELECT
    STRFTIME(datetime, '%Y-%m') AS month,
    AVG(Voltage) AS avg_voltage_v
FROM electric_power
WHERE STRFTIME(datetime, '%Y-%m') IN ('2007-06', '2007-07', '2007-08')
GROUP BY month
ORDER BY month;
