SELECT MAX(Global_intensity) AS value
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2010-07-12';
