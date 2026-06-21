SELECT COUNT(*) AS record_count
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2007-01-15'
  AND Sub_metering_1 > 0;
