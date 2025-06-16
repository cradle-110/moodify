import asyncio
import base64
from pathlib import Path
from playwright.async_api import async_playwright
from pymongo import MongoClient
from jinja2 import Template

mongo = MongoClient("mongodb://mongo:example@localhost:27017/")
saved_tracks = mongo.raw_data.saved_tracks

template = Template(Path("./ingestion/doc.html.jinja").read_text(encoding="utf-8"))

async def run():
    for i, saved_track in enumerate(saved_tracks.find()):
        if i % 100 == 0:
            print(f"Processing track {i + 1}")
        # HTML content with metadata and album art
        ## TODO update with other album arts
        images = [base64.b64encode(saved_track['album_art_data']).decode('utf-8')]
        html = template.render(
            genres=saved_track['artist_info']['genres'],
            song_name=saved_track['track']['name'],
            artists=[x['name'] for x in saved_track['track']['artists']],
            image_data=images,
        )
        async with async_playwright() as p:
            ## TODO don't repeat the browser launch for each song
            browser = await p.chromium.launch()
            page = await browser.new_page()
            # Serve the HTML from memory
            file_path = Path("temp_song_page.html")
            file_path.write_text(html, encoding="utf-8")

            await page.goto(file_path.absolute().as_uri())
            buffer = await page.screenshot(full_page=True)
            saved_tracks.update_one(
                {"_id": saved_track["_id"]},
                {"$set": {"document": buffer}}
            )

            await browser.close()
            file_path.unlink()  # clean up temporary HTML file

asyncio.run(run())
