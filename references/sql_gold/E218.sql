SELECT SUM(Sub_metering_3) AS total_wh
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2007-12-31';
