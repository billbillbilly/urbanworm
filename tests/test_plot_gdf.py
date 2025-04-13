from urbanworm import UrbanDataSet

def main():
    print("Step 1: Initialize dataset and load building footprints")
    # Smaller bounding box for quicker testing
    bbox = (-82.92575, 42.441675, -82.92525, 42.442025)
    data = UrbanDataSet()
    result = data.bbox2Buildings(bbox, source='osm', random_sample=5)
    print(result)
    print(f"Loaded {len(data.units)} buildings")

    print("\nStep 2: Define model and multiple prompts")
    system = '''
    Given a top view image, you are going to roughly estimate house conditions.
    Your answer should be based only on your observation.
    The format of your response must include question, answer (yes or no), explanation (within 50 words).
    '''
    prompt = {
        'top': '''
        Is there any damage on the roof?
        Is there a solar panel on the roof?
        '''
    }

    print("\nStep 3: Run inference using loopUnitChat")
    data.loopUnitChat(
        model='gemma3:12b',
        system=system,
        prompt=prompt,
        type='top',
        epsg=4326,  # WGS84 for visualization
        multi=False,
        output_gdf=False
    )

    print("\nStep 4: Convert results to GeoDataFrame")
    data.to_gdf()

    print("\nStep 5: Plot")
    data.plot_gdf()


if __name__ == "__main__":
    main()
