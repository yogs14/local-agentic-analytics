SELECT EXTRACT(YEAR FROM datetime) AS year, AVG(Voltage) AS avg_value
FROM electric_power
WHERE EXTRACT(MONTH FROM datetime) = 1
  AND EXTRACT(YEAR FROM datetime) IN (2007, 2008, 2009)
GROUP BY year
ORDER BY year;
