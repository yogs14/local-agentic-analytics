SELECT SUM(Sub_metering_1) AS total_wh
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2009-04-09';
