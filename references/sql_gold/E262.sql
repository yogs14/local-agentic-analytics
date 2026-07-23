SELECT datetime, Global_active_power
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2007
  AND Global_active_power IS NOT NULL
ORDER BY Global_active_power DESC, datetime ASC
LIMIT 1;
