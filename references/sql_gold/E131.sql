SELECT datetime, Global_intensity AS max_global_intensity_a
FROM electric_power
WHERE STRFTIME(datetime, '%Y-%m') = '2010-11'
  AND Global_intensity IS NOT NULL
ORDER BY Global_intensity DESC
LIMIT 1;
