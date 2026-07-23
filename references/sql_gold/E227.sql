SELECT MAX(Global_active_power) AS value
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2007
  AND EXTRACT(MONTH FROM datetime) = 7;
