SELECT SUM(Sub_metering_2) AS total_wh
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2010-01-04';
