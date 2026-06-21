SELECT SUM(Global_active_power) / 60.0 AS total_energy_kwh
FROM electric_power
WHERE STRFTIME(datetime, '%Y-%m') = '2008-08';
