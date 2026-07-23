SELECT MIN(Global_intensity) AS value
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2006-12-26';
