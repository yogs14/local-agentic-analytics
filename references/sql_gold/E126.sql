SELECT SUM(Sub_metering_3) AS total_sub_metering_3_wh
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2009-12-25';
