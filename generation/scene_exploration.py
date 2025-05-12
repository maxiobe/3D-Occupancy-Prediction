from truckscenes import TruckScenes

if __name__ == '__main__':

    truckscenes = TruckScenes(version='v1.0-trainval',
                              dataroot='/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval',
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

    i = 0
    for scene in truckscenes_test.scene:
        print(f"Scene {i}:")
        print(f"Scene name: {scene['name']}")
        print(f"Description: {scene['description']}")
        print()
        i += 1

    my_scene = truckscenes.scene[65]
    my_scene_token = my_scene['token']
    truckscenes.render_scene(my_scene_token)

    i = 0
    for scene in truckscenes.scene:
        print(f"Rendering scene {i}...")
        if i < 75:
            i += 1
            continue
        my_scene_token = scene['token']

        truckscenes.render_scene(my_scene_token)
        i += 1

