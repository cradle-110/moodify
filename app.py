import os
from spotipy import SpotifyOAuth, Spotify
from fastapi import FastAPI, Depends, Request
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
import uvicorn
import gradio as gr

from ingestion.liked_songs import save_user_saved_tracks
from ingestion.gen_docs import generate_docs
from ingestion.colqwen2_embeddings import embed_tracks
from storage.mongo import get_track_image, get_track_document
from models.colqwen2 import reranking_search_batch

app = FastAPI()

# Replace these with your own Spotify OAuth settings
SPOTIFY_CLIENT_ID = "c23563670ff943438fdc616383e9f0ea"
SPOTIFY_CLIENT_SECRET = "08e420d130d94312a20123663db0ec25"
SECRET_KEY = os.environ.get('SECRET_KEY') or "a_very_secret_key"

sp_oauth = SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri="http://127.0.0.1:8000/auth",
    scope="user-library-read,user-read-private"
)

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

def get_user(request: Request):
    user = request.session.get('user')
    if user:
        return user['display_name']
    return None

@app.route('/auth')
async def auth_callback(request: Request):
    code = request.query_params.get('code')
    token_info = sp_oauth.get_access_token(code)

    # Create Spotify client with user token
    sp = Spotify(auth=token_info['access_token'])

    # Retrieve current user details
    user_info = sp.current_user()

    request.session['user'] = user_info
    request.session['token_info'] = token_info
    return RedirectResponse(url='/')

@app.route('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    return RedirectResponse(url='/')

def check_user_login_status(request: gr.Request):
    logged_in = request.request.session.get('user') is not None
    if logged_in:
        welcome_message = f"Welcome back, {request.request.session['user']['display_name']}!"
    else:
        welcome_message = "Please log in to access your Spotify library."
    return (welcome_message, gr.update(visible=not logged_in), gr.update(visible=logged_in))

def import_tracks(num_tracks: int, import_all: bool, request: gr.Request, progress=gr.Progress()):
    # save tracks on mongo
    progress(0, desc="Starting, pulling user's saved tracks...")
    access_token = request.request.session.get('token_info')['access_token']
    if import_all:
        max_fetch = None
    else:
        max_fetch = num_tracks
    saved_ids = save_user_saved_tracks(Spotify(auth=access_token), max_fetch=max_fetch)
    # generate documents for tracks
    progress(1/3, desc="Tracks saved. Generating documents...")
    generate_docs(saved_ids)
    # generate embeddings for documents
    progress(2/3, desc="Documents generated. Computing embeddings...")
    embed_tracks(saved_ids)
    return len(saved_ids)

def lookup_song(prompt: str, search_limit: int, prefetch_limit: int):
    result = reranking_search_batch(prompt, search_limit=search_limit, prefetch_limit=prefetch_limit)
    images = []
    for point in result.points:
        caption = f"{point.payload['track_name']} - {point.score:.2f} - {point.payload['track_id']}"
        images.append((get_track_image(point.payload['track_id']), caption))
    return images

with gr.Blocks() as main_demo:
    images = gr.State([])
    m = gr.Markdown("Welcome to Gradio!") # gets overriden by greet
    with gr.Tab("Lookup Track"):
        with gr.Row():
            search_limit = gr.Slider(label="Search Limit", minimum=1, maximum=20, value=5, step=1)
            prefetch_limit = gr.Slider(label="Prefetch Limit", minimum=1, maximum=200, value=100, step=1)
        with gr.Row():
            prompt_input = gr.Textbox(label="Prompt", placeholder="Describe a track's album art")
            text_button = gr.Button("Search")
        with gr.Row():
            lookup_images = gr.Gallery(label="Search Results", scale=3, object_fit="contain")
            @gr.render(inputs=images)
            def render_captions(images):
                with gr.Column():
                    gr.Markdown("Captions:")
                    for i, (_, caption) in enumerate(images):
                        with gr.Row():
                            gr.Markdown(f"{i + 1}")
                            gr.Textbox(caption, show_label=False, scale=4)
        text_button.click(
            lookup_song,
            inputs=[prompt_input, search_limit, prefetch_limit],
            outputs=images
        )
        images.change(
            # grab the underlying PIL image and make the gallery caption the index
            lambda images: [ (img, f"{i + 1}") for i, (img, _) in enumerate(images) ],
            inputs=images,
            outputs=lookup_images
        )
    with gr.Tab("Import Library"):
        with gr.Row():
            num_tracks = gr.Number(label="Number of Tracks", value=10)
            import_all = gr.Checkbox(label="Import All Saved Tracks", value=False)
            num_imported = gr.Number(label="Number of Imported Tracks", value=0)
        import_button = gr.Button("Import")
        import_button.click(import_tracks, inputs=[num_tracks, import_all], outputs=num_imported)
    with gr.Tab("Document Lookup"):
        with gr.Row():
            track_id_input = gr.Textbox(label="Track ID", placeholder="Enter a track ID to view its document")
        doc_button = gr.Button("Get Document")
        doc_output = gr.Image(label="Document")
        doc_button.click(
            get_track_document,
            inputs=track_id_input,
            outputs=doc_output
        )
    with gr.Row():
        auth_url = sp_oauth.get_authorize_url()
        login_button = gr.Button("Login", link=auth_url)
        logout_button = gr.Button("Logout", link="/logout")
    main_demo.load(check_user_login_status, outputs=[m, login_button, logout_button])

app = gr.mount_gradio_app(app, main_demo, path="/")

if __name__ == '__main__':
    uvicorn.run(app)