from install_dense_lidar import dense_description

source = "<samples>10</samples><min_angle>-1.5707975</min_angle><max_angle>1.5707975</max_angle>"
result = dense_description(source)
assert "<samples>360</samples>" in result
assert "<min_angle>-3.14159265</min_angle>" in result
assert "<max_angle>3.14159265</max_angle>" in result
assert "<samples>10</samples>" not in result
print("DENSE_LIDAR_TRANSFORM_TEST_PASS")
