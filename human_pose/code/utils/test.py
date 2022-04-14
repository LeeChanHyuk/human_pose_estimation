from math import sin
from math import degrees
from math import radians

horizontal_angle = 85.2
vertical_angle = 58

x = (x-320)
y = (y-240)

detected_angle = (horizontal_angle/2) * (x/320)
new_x = sin(radians(detected_angle)) * depth_value

detected_angle = (vertical_angle/2) * (y/240)
new_y = sin(radians(detected_angle)) * depth_value