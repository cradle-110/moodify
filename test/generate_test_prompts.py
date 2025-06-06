from openai import OpenAI
from pymongo import MongoClient
import random
import string
import json

openai = OpenAI(api_key="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")

mongo = MongoClient("mongodb://mongo:example@localhost:27017/")
db = mongo.raw_data
saved_tracks = db.saved_tracks

requests = []
for track in saved_tracks.find():
    prompt = f"""we want to generate test prompts to gauge the quality of an application that uses ML to guess what song a user is describing, using context clues such as album art descriptions and general song metadata. You'll be provided with the album art and song information, and should generate 2 example prompts you think a user would use to get said song. The prompts should be short, one sentence at max, and should use things like color, art style type, visual composition, and prominent objects in the art, and sometimes include hints around things like artist name, genre (does not need to be the specific genre, just roughly), and the year / decade it came out. Prompts should be written like a user lazily doing a quick search - lowercase, short, and intending to convey semantic meaning, not necessarily correct grammar. The first prompt should be strictly about the album art, while the second can optionally include metadata hints as well. Example prompts:

    black and orange with monstercat icon
    red background with white flag
    eminem song with red curtains
    spinnin records song on black and white image
    black man silhouette with hands raised with blue text
    anime girl looking out at rain
    rap song with cyborg looking girl
    edm song with cherry blossom tree
    90s rock song with brown dog on the grass

    Your output should be concise, with just the two prompts separated by a new line. No prefixes like "1." or bullet points.

    The song for the provided album art is {track['track']['name']} by {track['track']['artists'][0]['name']}. genres include (if any): {track['artist_info']['genres']}"""
    #  
    input = [
        {"role": "user", "content": prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": track['track']['album']['images'][0]['url']
                }
            ]
        }
    ]
    # random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    requests.append({
        "custom_id": track['track']['id'],
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-4.1",
            "input": input
        }
    })

for batch, index in enumerate(range(0, len(requests), 500)):
    with open(f"batch_input-{batch}.jsonl", "w", encoding="utf-8") as f:
        for item in requests[index:index + 500]:
            f.write(json.dumps(item) + "\n")