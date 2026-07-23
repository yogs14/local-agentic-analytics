SELECT MAX(Global_intensity) AS value
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2008
  AND EXTRACT(MONTH FROM datetime) = 6;
