SELECT
    ISODOW(datetime) AS day_of_week,
    DAYNAME(datetime) AS day_name,
    AVG(Global_active_power) AS avg_global_active_power_kw
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2008
GROUP BY day_of_week, day_name
ORDER BY day_of_week;
