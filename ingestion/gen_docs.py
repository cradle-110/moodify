import base64
from pathlib import Path
from pymongo import MongoClient
from jinja2 import Template
from playwright.sync_api import sync_playwright

from storage.mongo import saved_tracks

template = Template(Path("./ingestion/doc.html.jinja").read_text(encoding="utf-8"))

def generate_docs(track_ids):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for i, track_id in enumerate(track_ids):
            if i % 100 == 0:
                print(f"Processing track {i + 1}")

            saved_track = saved_tracks.find_one({'track.id': track_id})
            # HTML content with metadata and album art
            ## TODO update with other album arts
            images = [base64.b64encode(saved_track['album_art_data']).decode('utf-8')]
            html = template.render(
                genres=saved_track['artist_info']['genres'],
                song_name=saved_track['track']['name'],
                artists=[x['name'] for x in saved_track['track']['artists']],
                image_data=images,
            )
            page = browser.new_page()
            # Serve the HTML from memory
            file_path = Path("temp_song_page.html")
            file_path.write_text(html, encoding="utf-8")

            page.goto(file_path.absolute().as_uri())
            buffer = page.screenshot(full_page=True)
            saved_tracks.update_one(
                {"_id": saved_track["_id"]},
                {"$set": {"document": buffer}}
            )

            file_path.unlink()  # clean up temporary HTML file
        browser.close()
