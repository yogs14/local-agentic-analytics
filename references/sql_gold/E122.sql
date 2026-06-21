SELECT
    EXTRACT(YEAR FROM datetime) AS year,
    AVG(Global_active_power) AS avg_global_active_power_kw
FROM electric_power
WHERE EXTRACT(MONTH FROM datetime) = 1
  AND EXTRACT(YEAR FROM datetime) IN (2007, 2008, 2009, 2010)
GROUP BY year
ORDER BY year;
