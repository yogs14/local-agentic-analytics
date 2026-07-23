SELECT AVG(Voltage) AS value
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2008
  AND EXTRACT(MONTH FROM datetime) = 8;
