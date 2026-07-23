SELECT CAST(datetime AS DATE) AS day, SUM(Global_active_power) / 60.0 AS total_energy_kwh
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2007
GROUP BY day
ORDER BY total_energy_kwh DESC, day ASC
LIMIT 5;
