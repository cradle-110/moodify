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

@app.get('/')
async def index(user: dict = Depends(get_user)):
    if user:
        return RedirectResponse(url='/gradio')
    else:
        auth_url = sp_oauth.get_authorize_url()
        return RedirectResponse(url=auth_url)

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
    return RedirectResponse(url='/gradio')

@app.route('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    return RedirectResponse(url='/')

with gr.Blocks() as login_demo:
    gr.Button("Login", link="/login")

app = gr.mount_gradio_app(app, login_demo, path="/login-demo")

def greet(request: gr.Request):
    return f"Welcome to Gradio, {request.username}"

def import_tracks(num_tracks: int, request: gr.Request, progress=gr.Progress()):
    # save tracks on mongo
    progress(0, desc="Starting, pulling user's saved tracks...")
    access_token = request.request.session.get('token_info')['access_token']
    saved_ids = save_user_saved_tracks(Spotify(auth=access_token), max_fetch=num_tracks)
    # generate documents for tracks
    progress(1/3, desc="Tracks saved. Generating documents...")
    generate_docs(saved_ids)
    # generate embeddings for documents
    progress(2/3, desc="Documents generated. Computing embeddings...")
    embed_tracks(saved_ids)
    return len(saved_ids)

with gr.Blocks() as main_demo:
    m = gr.Markdown("Welcome to Gradio!") # gets overriden by greet
    with gr.Tab("Lookup Track"):
        text_input = gr.Textbox()
        text_output = gr.Textbox()
        text_button = gr.Button("Flip")
    with gr.Tab("Import Library"):
        with gr.Row():
            num_tracks = gr.Number(label="Number of Tracks", value=10)
            num_imported = gr.Number(label="Number of Imported Tracks", value=0)
        import_button = gr.Button("Import")
        import_button.click(import_tracks, inputs=num_tracks, outputs=num_imported)
    with gr.Tab("Document Lookup"):
        doc_output = gr.Textbox(label="Document Content")
        doc_button = gr.Button("Process Document")
    gr.Button("Logout", link="/logout")
    main_demo.load(greet, None, m)

app = gr.mount_gradio_app(app, main_demo, path="/gradio", auth_dependency=get_user)

if __name__ == '__main__':
    uvicorn.run(app)