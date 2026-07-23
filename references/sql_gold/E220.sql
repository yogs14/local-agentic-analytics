SELECT MAX(Global_active_power) - MIN(Global_active_power) AS value_range
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2010-09-15';
