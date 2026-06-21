SELECT AVG(Global_active_power) AS avg_global_active_power_kw
FROM electric_power
WHERE STRFTIME(datetime, '%Y-%m') = '2007-01';
