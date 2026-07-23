SELECT datetime, Voltage
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2008
  AND Voltage IS NOT NULL
ORDER BY Voltage ASC, datetime ASC
LIMIT 1;
