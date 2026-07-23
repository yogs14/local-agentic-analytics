SELECT AVG(Global_intensity) AS value
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2010-09-14'
  AND EXTRACT(HOUR FROM datetime) BETWEEN 8 AND 10;
