SELECT AVG(Global_active_power) AS value
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2008-09-11'
  AND EXTRACT(HOUR FROM datetime) BETWEEN 10 AND 12;
