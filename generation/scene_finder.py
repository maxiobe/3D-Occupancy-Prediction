from truckscenes.truckscenes import TruckScenes

import os


def main():
    trucksc = TruckScenes(version='v1.0-trainval', dataroot='/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval', verbose=True)

    ### Tunnel
    # image_name = 'CAMERA_LEFT_FRONT_1692958612100411'
    image_name = 'CAMERA_LEFT_FRONT_1693828658701556'
    # image_name = 'CAMERA_LEFT_FRONT_1699974869900393'
    # image_name = 'CAMERA_LEFT_FRONT_1700579638900403'

    #### Tunnel und Regen
    # image_name = 'CAMERA_LEFT_FRONT_1699542503900955'

    #### Wenig Kontouren und schlechte Sicht
    # image_name = 'CAMERA_LEFT_FRONT_1697778762900744'

    #### Wenig Kontouren
    #image_name = 'CAMERA_LEFT_FRONT_1699009500301064'


    for idx in range(600):
        try:
            scene = trucksc.scene[idx]
            #print(f"Scene {idx}: {scene}")
            first_sample_token = scene['first_sample_token']
            my_sample = trucksc.get('sample',
                                    first_sample_token)
            #print(my_sample)
            sample_data = trucksc.get('sample_data', my_sample['data']['CAMERA_LEFT_FRONT'])
            # print(sample_data)
            while True:
                image_path = sample_data['filename']
                #print(image_path)
                filename = os.path.splitext(os.path.basename(image_path))[0]
                #print(filename)

                if filename == image_name:
                    print(f"Success: image found {image_name} in scene {idx}")
                    print(f"Scene {idx}: {scene}")
                    break

                if my_sample['next'] != '':
                    my_sample = trucksc.get('sample', my_sample['next'])
                    sample_data = trucksc.get('sample_data', my_sample['data']['CAMERA_LEFT_FRONT'])
                else:
                    break
        except IndexError:
            print(f"Index out of range {idx}")
            break


if __name__ == '__main__':
    main()