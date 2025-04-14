from urbanworm import UrbanDataSet


def main():
    print("Step 1: Load a small number of buildings")
    bbox = (-82.92575, 42.441675, -82.92525, 42.442025)
    data = UrbanDataSet()
    print(data.bbox2Buildings(bbox, source="osm", random_sample=3))

    print("\nStep 2: Run loopUnitChat to get LLM responses")
    system = '''
    You are a disaster inspector looking at top-down views of houses.
    Evaluate physical conditions only from what you see in the image.
    Each response must contain question, answer (yes/no), and a short explanation.
    '''
    prompt = {
        'top': '''
        Is there any roof damage?
        Does the roof appear clean or well-maintained?
        '''
    }

    data.loopUnitChat(
        model='gemma3:12b',
        system=system,
        prompt=prompt,
        type='top',
        epsg=4326,
        multi=False,
        output_gdf=False
    )
    data.to_gdf()

    print("\nStep 3: Use dataAnalyst to run summary-aware LLM reasoning")
    data.dataAnalyst("Based on the top_view_answer1 and top_view_answer2 columns, please summarize the counts of yes/no answers.")

    print("\n--- Message history preview ---")
    for msg in data.messageHistory:
        print(f"[{msg['role'].upper()}] {msg['content']}...\n")  # Print partial content


if __name__ == "__main__":
    main()
