from truckscenes import TruckScenes

def parse_description(desc):
    """
    Given a description like
      "weather.clear;area.terminal;daytime.morning;season.summer;lighting.illuminated;structure.regular;construction.unchanged"
    return a dict:
      {
        "weather": "clear",
        "area": "terminal",
        "daytime": "morning",
        "season": "summer",
        "lighting": "illuminated",
        "structure": "regular",
        "construction": "unchanged"
      }
    """
    parts = desc.split(';')
    parsed = {}
    for part in parts:
        if not part:
            continue
        key, val = part.split('.', 1)
        parsed[key] = val
    return parsed

def scene_finder_description(truckscenes):
    terminal_list = []
    terminal_list_id = []
    tunnel_list = []
    tunnel_list_id = []
    rain_list = []
    rain_list_id = []
    snow_list = []
    snow_list_id = []
    fog_list = []
    fog_list_id = []

    for i, scene in enumerate(truckscenes.scene):
        scene_description = scene['description']
        parsed = parse_description(scene_description)

        weather = parsed.get('weather')
        area = parsed.get('area')
        daytime = parsed.get('daytime')
        season = parsed.get('season')
        lighting = parsed.get('lighting')
        structure = parsed.get('structure')
        construction = parsed.get('construction')

        print(f"Scene #{i}:")
        print(f"  weather:      {weather}")
        print(f"  area:         {area}")
        print(f"  daytime:      {daytime}")
        print(f"  season:       {season}")
        print(f"  lighting:     {lighting}")
        print(f"  structure:    {structure}")
        print(f"  construction: {construction}")
        print()

        if area == 'terminal':
            terminal_list.append(scene['name'])
            terminal_list_id.append(i)
        if structure == 'tunnel':
            tunnel_list.append(scene['name'])
            tunnel_list_id.append(i)
        if weather == 'rain':
            rain_list.append(scene['name'])
            rain_list_id.append(i)
        if weather == 'snow':
            snow_list.append(scene['name'])
            snow_list_id.append(i)
        if weather == 'fog':
            fog_list.append(scene['name'])
            fog_list_id.append(i)

    print(f"{len(terminal_list)} scenes in terminal environment: {terminal_list}")
    print(f"Terminal scene ids: {terminal_list_id}")
    print(f"{len(tunnel_list)} scenes in tunnel environment: {tunnel_list}")
    print(f"Tunnel scene ids: {tunnel_list_id}")
    print(f"{len(rain_list)} scenes in rain environment: {rain_list}")
    print(f"Rain scene ids: {rain_list_id}")
    print(f"{len(snow_list)} scenes in snow environment: {snow_list}")
    print(f"Snow scene ids: {snow_list_id}")
    print(f"{len(fog_list)} scenes in fog environment: {fog_list}")
    print(f"Fog scene ids: {fog_list_id}")

def main():
    version = 'v1.0-mini'
    datapath = '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini'

    truckscenes = TruckScenes(version=version,
                              dataroot=datapath,
                              verbose=True)

    print("Mini scenes:")
    scene_finder_description(truckscenes)

    print()
    print()


    version = 'v1.0-trainval'
    datapath = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'

    truckscenes = TruckScenes(version=version,
                              dataroot=datapath,
                              verbose=True)

    print("Trainval scenes:")
    scene_finder_description(truckscenes)

    print()
    print()

    version = 'v1.0-test'
    datapath = '/home/max/ssd/Masterarbeit/TruckScenes/test/v1.0-test'

    truckscenes = TruckScenes(version=version,
                              dataroot=datapath,
                              verbose=True)

    print("Test scenes:")
    scene_finder_description(truckscenes)








if __name__ == "__main__":
    main()
