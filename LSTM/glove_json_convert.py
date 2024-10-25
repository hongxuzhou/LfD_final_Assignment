import json

def convert_glove_to_json(glove_file, json_file):
    """
    Convert GloVe file to JSON format.
    
    Args:
        glove_file: Path to the GloVe .txt file
        json_file: Path where to save the JSON file
    """
    print("Starting conversion...")
    embeddings = {}
    
    with open(glove_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:  # Progress indicator
                print(f"Processing line {i}")
                
            values = line.split()
            word = values[0]
            vector = [float(val) for val in values[1:]]
            embeddings[word] = vector
    
    print("Saving to JSON...")
    with open(json_file, 'w') as f:
        json.dump(embeddings, f)
    
    print(f"Done! Converted {len(embeddings)} words.")

# Use the function
glove_file = "/Users/hongxuzhou/Desktop/lfd_final_glove/original/glove.twitter.27B.100d.txt"  # Your input file
json_file = "/Users/hongxuzhou/Desktop/lfd_final_glove/converted_json/glove_twitter_100d.json"      # Your output file
convert_glove_to_json(glove_file, json_file)