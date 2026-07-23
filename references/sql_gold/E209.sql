SELECT MIN(Global_active_power) AS value
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2008-02-29';
