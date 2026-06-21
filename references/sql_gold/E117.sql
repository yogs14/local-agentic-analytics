SELECT AVG(Voltage) AS avg_voltage_v
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2006-12-16'
  AND Global_intensity > 10;
