SELECT
    EXTRACT(WEEK FROM datetime) AS iso_week,
    SUM(Global_active_power) / 60.0 AS total_energy_kwh
FROM electric_power
WHERE STRFTIME(datetime, '%Y-%m') = '2007-01'
GROUP BY iso_week
ORDER BY iso_week;
