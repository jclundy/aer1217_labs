import numpy as np
mass = 29.0 / 1000 # 29 grams
g = 9.81
weight = mass*g
payload_mass = 15/1000
payload_weight = payload_mass * g
max_combined_thrust = weight + payload_weight # additional 15g payload

max_lateral_thrust = np.sqrt(max_combined_thrust**2 - weight**2)

max_lateral_acceleration = max_lateral_thrust / mass

max_tilt_angle = np.arctan2(weight, max_lateral_acceleration) * 180/np.pi

print("Weight={:.3f} N, Combined Thrust={:.3f} N".format(weight, max_combined_thrust))
print("Max lateral thrust={:.3f} N".format(max_lateral_thrust))
print("Max acceleration thrust={:.3f} m/s^2".format(max_lateral_acceleration))
print("max tilt angle={:.2f}".format(max_tilt_angle))

# Brushless max speed: 2.5 m/s
# DC coreless max speed: 1 m/s
