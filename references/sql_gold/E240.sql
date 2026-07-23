SELECT COUNT(*) AS record_count
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2008
  AND EXTRACT(MONTH FROM datetime) = 7
  AND Voltage > 241.0;
