SELECT
    EXTRACT(HOUR FROM datetime) AS hour_of_day,
    AVG(Global_active_power) AS avg_global_active_power_kw
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2008
GROUP BY hour_of_day
ORDER BY avg_global_active_power_kw DESC
LIMIT 3;
