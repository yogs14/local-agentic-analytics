SELECT AVG(Global_active_power) AS value
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2008-01-19'
  AND EXTRACT(HOUR FROM datetime) BETWEEN 6 AND 8;
