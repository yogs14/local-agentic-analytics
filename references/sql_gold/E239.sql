SELECT COUNT(*) AS record_count
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2010
  AND EXTRACT(MONTH FROM datetime) = 5
  AND Global_active_power > 2.5;
