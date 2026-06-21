SELECT COUNT(*) FILTER (WHERE Voltage IS NULL) AS missing_voltage_count
FROM electric_power;
