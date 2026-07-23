SELECT datetime, Global_intensity
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2009
  AND EXTRACT(MONTH FROM datetime) = 11
  AND Global_intensity IS NOT NULL
ORDER BY Global_intensity DESC, datetime ASC
LIMIT 1;
