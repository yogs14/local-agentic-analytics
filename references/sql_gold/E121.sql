SELECT datetime, Global_active_power AS max_global_active_power_kw
FROM electric_power
ORDER BY Global_active_power DESC
LIMIT 1;
