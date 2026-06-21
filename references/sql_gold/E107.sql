SELECT SUM(Global_active_power) / 60.0 AS total_energy_kwh
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2007-01-20'
  AND EXTRACT(HOUR FROM datetime) = 19;
