SELECT MAX(Voltage) - MIN(Voltage) AS value_range
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2007-08-11';
