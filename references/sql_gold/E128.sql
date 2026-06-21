SELECT QUANTILE_CONT(Voltage, 0.5) AS median_voltage_v
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2010-07-14';
