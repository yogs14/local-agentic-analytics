SELECT 100.0 * SUM(CASE WHEN Global_active_power > 0.6 THEN 1 ELSE 0 END) / COUNT(*) AS pct
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2009
  AND EXTRACT(MONTH FROM datetime) = 5;
