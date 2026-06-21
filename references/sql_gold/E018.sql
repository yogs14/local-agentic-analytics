SELECT
    COUNT(*) FILTER (WHERE Global_active_power IS NULL) AS missing_global_active_power_count
FROM electric_power;
