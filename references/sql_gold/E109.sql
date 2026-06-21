SELECT
    EXTRACT(HOUR FROM datetime) AS hour_of_day,
    AVG(Global_active_power) AS avg_global_active_power_kw
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2007-02-01'
GROUP BY hour_of_day
ORDER BY hour_of_day;
