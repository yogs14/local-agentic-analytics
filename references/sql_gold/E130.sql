SELECT datetime, Voltage AS min_voltage_v
FROM electric_power
WHERE Voltage IS NOT NULL
ORDER BY Voltage ASC
LIMIT 1;
