SELECT AVG(Voltage) AS avg_voltage_v
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2009-07-04';
