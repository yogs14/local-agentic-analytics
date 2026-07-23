SELECT AVG(Voltage) AS value
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2009-12-05';
