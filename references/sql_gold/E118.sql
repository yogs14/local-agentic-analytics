SELECT COUNT(*) AS record_count
FROM electric_power
WHERE STRFTIME(datetime, '%Y-%m') = '2007-12'
  AND Sub_metering_3 > 0;
