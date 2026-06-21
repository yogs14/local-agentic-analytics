SELECT COUNT(*) AS record_count
FROM electric_power
WHERE Global_active_power > 5
  AND Voltage < 235;
