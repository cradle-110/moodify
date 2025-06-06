import os
import json

art_prompts = []
enriched_prompts = []

# Loop through all relevant .jsonl files
for filename in os.listdir('./test/batch_outputs'):
    with open(f"./test/batch_outputs/{filename}", 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                content = data["response"]["body"]["output"][0]["content"][0]["text"]
                track_id = data["custom_id"]
                prompts = content.strip().split('\n', 1)
                if len(prompts) == 2:
                    art_prompts.append({
                        "track_id": track_id,
                        "prompt": prompts[0].strip()
                    })
                    enriched_prompts.append({
                        "track_id": track_id,
                        "prompt": prompts[1].strip()
                    })
                else:
                    print(f"Skipping incomplete prompt in file {filename}")
            except Exception as e:
                print(f"Error processing line in {filename}: {e}")

# Write output JSON files
with open("./test/art_prompts.json", "w", encoding="utf-8") as f:
    json.dump(art_prompts, f, indent=2, ensure_ascii=False)

with open("./test/enriched_prompts.json", "w", encoding="utf-8") as f:
    json.dump(enriched_prompts, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(art_prompts)} prompts to 'art_prompts.json' and 'enriched_prompts.json'")
