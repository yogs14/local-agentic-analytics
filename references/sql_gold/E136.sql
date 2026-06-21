SELECT MAX(Voltage) - MIN(Voltage) AS voltage_range_v
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2009-07-04';
