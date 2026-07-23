SELECT AVG(Sub_metering_1) AS avg_sub1, AVG(Sub_metering_2) AS avg_sub2, AVG(Sub_metering_3) AS avg_sub3
FROM electric_power
WHERE EXTRACT(YEAR FROM datetime) = 2007
  AND EXTRACT(MONTH FROM datetime) = 1;
