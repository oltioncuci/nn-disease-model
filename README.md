    # test_scenarios = [
    #     {"name": "🌟 SCENARIO: Optimal", "msg": "Target: 85-100", "data": {"month": 10, "air_temp_day_avg": 18.5, "air_temp_night_avg": 14.0, "soil_temp_avg": 17.5, "soil_moisture_avg": 70.0, "air_humidity_avg": 90.0, "wind_speed_avg": 1.5, "rainfall_3d_total": 15.0, "rainfall_7d_total": 25.0, "rain_days_7d": 5, "max_daily_rain_7d": 8.0}},
    #     {"name": "🔥 SCENARIO: Heat", "msg": "Target: Porcini crash", "data": {"month": 7, "air_temp_day_avg": 31.0, "air_temp_night_avg": 20.0, "soil_temp_avg": 25.5, "soil_moisture_avg": 55.0, "air_humidity_avg": 60.0, "wind_speed_avg": 4.0, "rainfall_3d_total": 10.0, "rainfall_7d_total": 30.0, "rain_days_7d": 3, "max_daily_rain_7d": 15.0}},
    #     {"name": "🌊 SCENARIO: Saturation", "msg": "Target: Porcini drop", "data": {"month": 9, "air_temp_day_avg": 17.0, "air_temp_night_avg": 13.0, "soil_temp_avg": 16.0, "soil_moisture_avg": 92.0, "air_humidity_avg": 95.0, "wind_speed_avg": 1.0, "rainfall_3d_total": 40.0, "rainfall_7d_total": 65.0, "rain_days_7d": 6, "max_daily_rain_7d": 20.0}},
    #     {"name": "❄️ SCENARIO: Cold", "msg": "Target: Both < 15", "data": {"month": 11, "air_temp_day_avg": 11.5, "air_temp_night_avg": 6.0, "soil_temp_avg": 10.5, "soil_moisture_avg": 35.0, "air_humidity_avg": 50.0, "wind_speed_avg": 3.5, "rainfall_3d_total": 0.0, "rainfall_7d_total": 5.0, "rain_days_7d": 1, "max_daily_rain_7d": 5.0}},
    #     {"name": "🍄 CASE 1: Porcini High", "msg": "Target: Porcini Sweet spot", "data": {"month": 10, "air_temp_day_avg": 16.0, "air_temp_night_avg": 12.0, "soil_temp_avg": 15.0, "soil_moisture_avg": 70.0, "air_humidity_avg": 92.0, "wind_speed_avg": 1.2, "rainfall_3d_total": 15.0, "rainfall_7d_total": 22.0, "rain_days_7d": 5, "max_daily_rain_7d": 6.0}},
    #     {"name": "💨 SCENARIO: Wind", "msg": "Target: Desiccation", "data": {"month": 6, "air_temp_day_avg": 20.0, "air_temp_night_avg": 14.0, "soil_temp_avg": 18.0, "soil_moisture_avg": 65.0, "air_humidity_avg": 70.0, "wind_speed_avg": 8.0, "rainfall_3d_total": 12.0, "rainfall_7d_total": 20.0, "rain_days_7d": 4, "max_daily_rain_7d": 7.0}}
    # ]
#     test_scenarios = [
#     {
#         "name": "❄️ THE COLD SURVIVOR",
#         "desc": "Late Oct: 12°C. Below Porcini floor, but Chanterelle 'might' still fruit.",
#         "data": {
#             "month": 10, "air_temp_day_avg": 12.0, "air_temp_night_avg": 7.0,
#             "soil_temp_avg": 11.0, "soil_moisture_avg": 75.0, "air_humidity_avg": 85.0,
#             "wind_speed_avg": 2.0, "rainfall_3d_total": 10.0, "rainfall_7d_total": 20.0,
#             "rain_days_7d": 3, "max_daily_rain_7d": 10.0
#         }
#     },
#     {
#         "name": "🔥 THE SUMMER SHRIVEL",
#         "desc": "Hot July: 27°C. Chanterelles shrivel in heat; Porcini might hold on slightly longer.",
#         "data": {
#             "month": 7, "air_temp_day_avg": 27.0, "air_temp_night_avg": 19.0,
#             "soil_temp_avg": 22.0, "soil_moisture_avg": 55.0, "air_humidity_avg": 60.0,
#             "wind_speed_avg": 4.0, "rainfall_3d_total": 2.0, "rainfall_7d_total": 5.0,
#             "rain_days_7d": 1, "max_daily_rain_7d": 5.0
#         }
#     },
#     {
#         "name": "🌊 THE DROWNED FOREST",
#         "desc": "Massive Flooding: 90mm rain. Soil saturation is too high for Porcini.",
#         "data": {
#             "month": 9, "air_temp_day_avg": 18.0, "air_temp_night_avg": 12.0,
#             "soil_temp_avg": 16.0, "soil_moisture_avg": 98.0, "air_humidity_avg": 95.0,
#             "wind_speed_avg": 1.0, "rainfall_3d_total": 60.0, "rainfall_7d_total": 95.0,
#             "rain_days_7d": 5, "max_daily_rain_7d": 50.0
#         }
#     },
#     {
#         "name": "🌵 THE FALSE AUTUMN",
#         "desc": "Perfect temp (18°C) but 0 rain for 7 days. Score should be near 0.",
#         "data": {
#             "month": 9, "air_temp_day_avg": 18.0, "air_temp_night_avg": 11.0,
#             "soil_temp_avg": 15.0, "soil_moisture_avg": 30.0, "air_humidity_avg": 45.0,
#             "wind_speed_avg": 3.0, "rainfall_3d_total": 0.0, "rainfall_7d_total": 0.0,
#             "rain_days_7d": 0, "max_daily_rain_7d": 0.0
#         }
#     }
# ]