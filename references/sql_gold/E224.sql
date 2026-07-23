SELECT AVG(Global_active_power) AS value
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2010
  AND EXTRACT(MONTH FROM datetime) = 4;
