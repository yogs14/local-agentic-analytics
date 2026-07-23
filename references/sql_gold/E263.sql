WITH daily AS (
    SELECT CAST(datetime AS DATE) AS day,
           SUM(Global_active_power) / 60.0 AS energy_kwh
    FROM electric_power
    WHERE EXTRACT(YEAR FROM datetime) = 2008
      AND EXTRACT(MONTH FROM datetime) = 10
    GROUP BY day
)
SELECT day, energy_kwh - LAG(energy_kwh) OVER (ORDER BY day) AS delta_kwh
FROM daily
QUALIFY delta_kwh IS NOT NULL
ORDER BY delta_kwh DESC, day ASC
LIMIT 1;
