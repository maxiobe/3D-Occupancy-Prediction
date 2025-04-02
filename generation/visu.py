from truckscenes import TruckScenes

trucksc = TruckScenes('v1.0-trainval', '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval', True)

my_scene = trucksc.scene[1]
print(my_scene)

first_sample_token = my_scene['first_sample_token']
# print(first_sample_token)
# trucksc.render_sample(first_sample_token)

my_sample = trucksc.get('sample', first_sample_token)
print(my_sample)

print(my_sample['data'])

sensor = 'CAMERA_LEFT_FRONT'
cam_front_data = trucksc.get('sample_data', my_sample['data'][sensor])
print(cam_front_data)

# trucksc.render_sample_data(cam_front_data['token'])