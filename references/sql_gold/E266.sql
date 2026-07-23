SELECT COUNT(DISTINCT CAST(datetime AS DATE)) AS unique_days
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2007;
