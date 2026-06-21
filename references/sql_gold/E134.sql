SELECT
    100.0 * COUNT(*) FILTER (WHERE Global_active_power > 4) / COUNT(*)
        AS pct_minutes_above_4kw
FROM electric_power
WHERE STRFTIME(datetime, '%Y-%m') = '2007-01';
