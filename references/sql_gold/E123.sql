SELECT
    EXTRACT(YEAR FROM datetime) AS year,
    SUM(Global_active_power) / 60.0 AS total_energy_kwh
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) IN (2007, 2008, 2009)
GROUP BY year
ORDER BY year;
