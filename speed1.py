import math
import numpy as np
import cv2
from tensorflow.keras.models import load_model

class SpeedRecommender:
    def __init__(self):
        self.previous_speed = 0  # For smoothing filter
        self.road_condition_model = load_model("#dataset_name")  # Pre-trained ML model for road condition

    def calculate_curvature_radius(self, lane_points):
        # Using lane points detected by camera to calculate curvature radius
        x = np.array([p[0] for p in lane_points])
        y = np.array([p[1] for p in lane_points])

        # Fit a parabola (second-degree polynomial) to the lane points
        fit = np.polyfit(x, y, 2)
        a = fit[0]

        # Curvature radius formula: R = (1 + (2ax + b)^2)^(3/2) / |2a|
        curvature_radius = (1 + (2 * a * x.mean())**2)**(3/2) / abs(2 * a)
        return curvature_radius

    def analyze_road_surface(self, frame):
        # Preprocess the frame for the ML model
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        input_data = np.expand_dims(normalized_frame, axis=0)

        # Predict road condition
        prediction = self.road_condition_model.predict(input_data)
        wet_surface_probability = prediction[0][0]  # Assuming binary classification: [wet, dry]

        return wet_surface_probability > 0.5  # Return True if wet surface detected

    def calculate_friction_coefficient(self, frame):
        # Estimate friction coefficient based on visual cues from camera
        is_wet = self.analyze_road_surface(frame)
        if is_wet:
            return 0.6  # Typical value for wet asphalt
        else:
            return 0.8  # Typical value for dry asphalt

    def calculate_cross_slope(self, camera_data):
        # Estimate cross slope based on road tilt detected by the camera
        return camera_data.get("road_tilt", 2)  # Default tilt value in degrees

    def road_geometry_speed(self, curvature_radius, friction_coefficient, cross_slope):
        # Speed based on road geometry
        return math.sqrt(127 * curvature_radius * (friction_coefficient + 0.001 * cross_slope))

    def environmental_adjustment(self, weather_condition, pavement_condition, visibility_conditions):
        # Fuzzy adjustments based on environment
        adjustment_factor = 1.0  # Default factor

        # Weather conditions
        if weather_condition == "rainy":
            adjustment_factor -= 0.2
        if pavement_condition == "wet":
            adjustment_factor -= 0.2

        # Visibility conditions
        light_condition = visibility_conditions.get("light_condition", "daylight")
        fog = visibility_conditions.get("fog", "high")
        temperature = visibility_conditions.get("temperature", 10)
        humidity = visibility_conditions.get("humidity", 50)

        if light_condition == "low light":
            adjustment_factor -= 0.1
        elif light_condition == "night":
            adjustment_factor -= 0.2

        if fog == "medium":
            adjustment_factor -= 0.2
        elif fog == "low":
            adjustment_factor -= 0.4

        if temperature <= 5 and temperature > 0:
            adjustment_factor -= 0.1
        elif temperature <= 0:
            adjustment_factor -= 0.2

        if humidity > 60:
            adjustment_factor -= 0.1

        return max(adjustment_factor, 0.5)  # Ensure a minimum adjustment factor

    def driver_preference_adjustment(self, comfort_level):
        # Adjust speed based on driver comfort level
        preferences = {
            "calm": 0.8,
            "normal": 1.0,
            "urgent": 1.2,
            "emergency": 1.5
        }
        return preferences.get(comfort_level, 1.0)

    def safety_check(self, curvature_radius, friction_coefficient, track_width, center_of_gravity_height):
        # Calculate safety-related speeds
        sliding_speed = math.sqrt(friction_coefficient * 9.81 * curvature_radius)
        overturning_speed = math.sqrt((track_width * curvature_radius) / (2 * center_of_gravity_height))

        return min(sliding_speed, overturning_speed)

    def camera_gps_adjustment(self, camera_data, gps_data):
        # Adjustment based on detected road conditions from camera and GPS
        adjustment_factor = 1.0

        # Camera-based detections
        if camera_data.get("sharp_turn", False):
            adjustment_factor -= 0.2
        if camera_data.get("narrow_lane", False):
            adjustment_factor -= 0.1
        if camera_data.get("obstacle_detected", False):
            adjustment_factor -= 0.3

        # GPS-based data
        if gps_data.get("road_type", "highway") == "residential":
            adjustment_factor -= 0.2
        if gps_data.get("speed_limit", 100) < 50:
            adjustment_factor *= 0.8

        return max(adjustment_factor, 0.5)

    def smoothing_filter(self, recommended_speed, alpha=0.6):
        # Smooth the speed using exponential smoothing
        smoothed_speed = alpha * recommended_speed + (1 - alpha) * self.previous_speed
        self.previous_speed = smoothed_speed
        return smoothed_speed

    def recommend_speed(self, road_params, env_params, driver_params, visibility_conditions, camera_data, gps_data, frame):
        # Calculate parameters using camera and GPS
        curvature_radius = self.calculate_curvature_radius(camera_data["lane_points"])
        friction_coefficient = self.calculate_friction_coefficient(frame)
        cross_slope = self.calculate_cross_slope(camera_data)

        weather_condition = env_params["weather"]
        pavement_condition = env_params["pavement"]

        comfort_level = driver_params["comfort_level"]

        # Step 1: Road geometry speed
        geo_speed = self.road_geometry_speed(curvature_radius, friction_coefficient, cross_slope)

        # Step 2: Environmental adjustment
        env_adjust = self.environmental_adjustment(weather_condition, pavement_condition, visibility_conditions)

        # Step 3: Driver preference adjustment
        driver_adjust = self.driver_preference_adjustment(comfort_level)

        # Step 4: Camera and GPS adjustment
        cam_gps_adjust = self.camera_gps_adjustment(camera_data, gps_data)

        # Combine the adjusted speed
        adjusted_speed = geo_speed * env_adjust * driver_adjust * cam_gps_adjust

        # Step 5: Safety checks
        safety_speed = self.safety_check(
            curvature_radius, friction_coefficient,
            road_params["track_width"], road_params["center_of_gravity_height"]
        )

        final_speed = min(adjusted_speed, safety_speed)

        # Step 6: Apply smoothing filter
        smoothed_speed = self.smoothing_filter(final_speed)

        return smoothed_speed

# Example usage
road_params = {
    "track_width": 1.8,
    "center_of_gravity_height": 0.5
}

env_params = {
    "weather": "rainy",
    "pavement": "wet"
}

driver_params = {
    "comfort_level": "normal"
}

visibility_conditions = {
    "light_condition": "night",
    "fog": "medium",
    "temperature": 2,
    "humidity": 70
}

camera_data = {
    "lane_points": [(0, 0), (10, 2), (20, 8)],
    "road_tilt": 2,
    "sharp_turn": True,
    "narrow_lane": False,
    "obstacle_detected": False
}

gps_data = {
    "road_type": "residential",
    "speed_limit": 40
}

# Simulate a frame capture from a camera
frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder for actual camera frame

recommender = SpeedRecommender()
recommended_speed = recommender.recommend_speed(road_params, env_params, driver_params, visibility_conditions, camera_data, gps_data, frame)
print(f"Recommended Speed: {recommended_speed:.2f} km/h")
