SELECT AVG(Global_intensity) AS value
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2009-11-07';
