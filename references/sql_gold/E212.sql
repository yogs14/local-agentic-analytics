SELECT MIN(Global_reactive_power) AS value
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2010-09-07';
