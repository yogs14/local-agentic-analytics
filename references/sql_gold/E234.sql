SELECT AVG(Voltage) AS value
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2009-06-16'
  AND EXTRACT(HOUR FROM datetime) BETWEEN 8 AND 11;
