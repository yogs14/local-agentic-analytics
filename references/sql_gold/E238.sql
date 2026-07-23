SELECT COUNT(*) AS record_count
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2007-05-28'
  AND Global_intensity > 10.0;
