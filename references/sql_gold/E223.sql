SELECT STDDEV_SAMP(Global_active_power) AS value
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2007-03-22';
