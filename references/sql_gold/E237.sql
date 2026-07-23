SELECT COUNT(*) AS record_count
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2007-02-07'
  AND Global_active_power > 2.5;
