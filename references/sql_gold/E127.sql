SELECT STDDEV_SAMP(Global_active_power) AS stddev_global_active_power_kw
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2010-07-14';
