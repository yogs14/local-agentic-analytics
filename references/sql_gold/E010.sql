SELECT
    COUNT(*) AS record_count
FROM electric_power
WHERE CAST(datetime AS DATE) = DATE '2006-12-16';
