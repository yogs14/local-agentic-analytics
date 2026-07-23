SELECT AVG(Global_intensity) AS value
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2007
  AND EXTRACT(MONTH FROM datetime) = 5;
