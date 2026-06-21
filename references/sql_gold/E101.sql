SELECT AVG(Global_active_power) AS avg_global_active_power_kw
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2007-01-15';
