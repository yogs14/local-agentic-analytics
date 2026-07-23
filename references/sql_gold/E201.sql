SELECT AVG(Global_active_power) AS value
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2008-06-18';
