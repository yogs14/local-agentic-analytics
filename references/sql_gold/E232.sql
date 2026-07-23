SELECT SUM(Global_active_power) / 60.0 AS total_energy_kwh
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2008
  AND EXTRACT(MONTH FROM datetime) = 1;
