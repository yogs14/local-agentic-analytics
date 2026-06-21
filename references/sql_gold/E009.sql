SELECT
    DATE_TRUNC('hour', datetime) AS hour_start,
    SUM(Global_active_power) AS total_global_active_power_kw
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2006-12-16'
GROUP BY hour_start
ORDER BY total_global_active_power_kw DESC
LIMIT 1;
