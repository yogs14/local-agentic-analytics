SELECT
    COUNT(*) FILTER (WHERE Global_active_power IS NULL) AS missing_global_active_power_count,
    COUNT(*) FILTER (WHERE Global_reactive_power IS NULL) AS missing_global_reactive_power_count,
    COUNT(*) FILTER (WHERE Voltage IS NULL) AS missing_voltage_count,
    COUNT(*) FILTER (WHERE Global_intensity IS NULL) AS missing_global_intensity_count,
    COUNT(*) FILTER (WHERE Sub_metering_1 IS NULL) AS missing_sub_metering_1_count,
    COUNT(*) FILTER (WHERE Sub_metering_2 IS NULL) AS missing_sub_metering_2_count,
    COUNT(*) FILTER (WHERE Sub_metering_3 IS NULL) AS missing_sub_metering_3_count
FROM electric_power;
