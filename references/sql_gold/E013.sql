SELECT
    AVG(Sub_metering_3) AS avg_sub_metering_3_wh
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2006-12-16';
