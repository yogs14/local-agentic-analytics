SELECT AVG(Global_intensity) AS avg_global_intensity_a
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2010-03-21';
