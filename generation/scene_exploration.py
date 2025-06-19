from truckscenes import TruckScenes

if __name__ == '__main__':

    version = 'v1.0-trainval'
    # version = 'v1.0-mini'
    datapath = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'
    # datapath = '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini'
    truckscenes = TruckScenes(version=version,
                              dataroot=datapath,
                              verbose=True)

    # truckscenes.list_scenes()

    print("Trainval scenes:")
    print()

    i = 0
    for scene in truckscenes.scene:
        print(f"Scene {i}:")
        print(f"Scene name: {scene['name']}")
        print(f"Description: {scene['description']}")
        print()
        i += 1

    truckscenes_test = TruckScenes(version='v1.0-test',
                              dataroot='/home/max/ssd/Masterarbeit/TruckScenes/test/v1.0-test',
                              verbose=True)

    print("Test scenes:")
    print()

    for scene in truckscenes_test.scene:
        print(f"Scene {i}:")
        print(f"Scene name: {scene['name']}")
        print(f"Description: {scene['description']}")
        print()

    """# scene_list = [3, 46, 47, 53, 54, 107, 108, 109, 110, 114, 115, 116]
    scene_list = [46, 47, 107, 108, 109, 110]
    for scene_id in scene_list:
        print(f"Rendering single scene {scene_id}...")
        my_scene = truckscenes_test.scene[scene_id]
        my_scene_token = my_scene['token']
        truckscenes_test.render_scene(my_scene_token)"""


    # scene_list = [1, 2, 4, 44, 45, 46, 64, 65, 66, 68, 69, 70, 139, 140, 203, 204, 205, 206, 239, 240, 241, 257, 258, 259, 260, 272, 273, 419, 420, 421, 423, 447, 448, 449, 450, 451, 452, 453, 454, 455, 457, 458, 459, 460, 461, 462, 463, 492, 517, 518, 519, 520, 597]
    # scene_list = [2, 65, 139, 140, 240, 419, 420, 421, 447, 448, 449, 450, 451, 452, 454, 455, 457, 458, 459, 460, 461, 462, 463, 517, 518, 519, 520]
    scene_list = [5]
    for scene_id in scene_list:
        print(f"Rendering single scene {scene_id}...")
        my_scene = truckscenes.scene[scene_id]
        my_scene_token = my_scene['token']
        truckscenes.render_scene(my_scene_token)

    i = 0
    for scene in truckscenes.scene:
        print(f"Rendering scene {i}...")
        #if i < 75:
         #   i += 1
          #  continue
        my_scene_token = scene['token']

        truckscenes.render_scene(my_scene_token)
        i += 1

