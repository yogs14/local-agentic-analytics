SELECT
    AVG(Global_active_power) AS avg_global_active_power_kw
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2006-12-16';
