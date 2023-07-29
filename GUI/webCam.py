import cv2
from dash import Dash, dcc, html, Input, Output, State, no_update, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
import io
import base64
from flask import Flask, Response


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(base64_string):
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes))


def predict(image):
    image.save('cropped_img.png')
    return 'class predicted'


server = Flask(__name__)


@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


app = Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ], server=server)
# Constants
img_width = 1100
img_height = 650
scale_factor = 0.5

video_modal = html.Div(
    dbc.Modal(
        [
            dbc.ModalHeader("Video Feed"),
            dbc.ModalBody(html.Img(id="video-feed", src="/video_feed", width=450, height=250)),
            dbc.ModalFooter(dbc.Button("Take Photo", id="take-photo-btn", color="primary", className="mr-2"),)
        ],
        id="video-modal",
        size="md",
    )
)

app.layout = dbc.Container([
    dcc.Upload(
        id='upload-image',
        children=dbc.Container([
            'Drag and Drop or ',
            dbc.Button('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    dbc.Button('Use Camera', id='open-modal-btn'),
    video_modal,
    dcc.Store(id='cropped-image-store', data=None),  # Store the cropped image data
    dbc.Row([
        dbc.Col(
            dbc.Container(id='output-image-upload'),
            width=6
        ),
        dbc.Col(
            dbc.Container(id="cropped-image-container"),
            width=6
        ),
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Button("Crop Image", id="crop-button", style={'margin': '10px'}),
            width=6
        ),
        dbc.Col(
            dbc.Button("Predict Image", id="predict-button", style={'margin': '10px'}),
            width=6
        ),
    ]),
    html.H3(id="predicted-class")
], fluid=True)


def parse_contents(fig, filename):
    return dbc.Card([
        html.H5(filename),
        dcc.Graph(figure=fig,
                  config={'displayModeBar': False},  # Disable the mode bar
                  style={
                      'width': '100%',
                      'height': '100%',  # Take full height of the card
                      'padding': 0,  # Remove padding to fit the figure completely
                  }, ),
    ],
        style={
            'width': '100%',
            'height': '400px',
            'textAlign': 'center',
            'padding': '20px'
        },
    )


@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents'),
    Input("take-photo-btn", "n_clicks"),
    State('upload-image', 'filename')
)
def update_output(list_of_contents, take_photo_clicks, list_of_names):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Handle file upload
    if trigger_id == 'upload-image':
        fig = go.Figure()

        # Configure axes
        fig.update_xaxes(
            visible=False,
            range=[0, img_width * scale_factor]
        )

        fig.update_yaxes(
            visible=False,
            range=[0, img_height * scale_factor],
            scaleanchor="x"
        )

        if list_of_contents:
            fig.add_layout_image(
                dict(
                    x=0,
                    sizex=img_width * scale_factor,
                    y=img_height * scale_factor,
                    sizey=img_height * scale_factor,
                    xref="x",
                    yref="y",
                    opacity=1.0,
                    layer="below",
                    sizing="stretch",
                    source=list_of_contents)
            )

        fig.update_layout(
            width=img_width * scale_factor,
            height=img_height * scale_factor,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )

        children = [parse_contents(fig, list_of_names)]

        return children

    # Handle taking a photo from the camera
    elif trigger_id == 'take-photo-btn':
        if take_photo_clicks is None or take_photo_clicks == 0:
            raise PreventUpdate

        # Take a photo from the camera
        camera = VideoCamera()
        frame = camera.get_frame()
        fig = go.Figure()

        # Configure axes
        fig.update_xaxes(
            visible=False,
            range=[0, img_width * scale_factor]
        )

        fig.update_yaxes(
            visible=False,
            range=[0, img_height * scale_factor],
            scaleanchor="x"
        )

        fig.add_layout_image(
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source="data:image/jpeg;base64," + base64.b64encode(frame).decode("utf-8"))
            )

        fig.update_layout(
            width=img_width * scale_factor,
            height=img_height * scale_factor,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )

        children = [parse_contents(fig, 'from webcam')]
        return children

    return no_update




@app.callback(
    Output("cropped-image-container", "children"),
    Output("cropped-image-store", "data"),  # Store the cropped image data
    Input("crop-button", "n_clicks"),
    State("output-image-upload", "children"),
)
def crop_image(n_clicks, children):
    if n_clicks and children:
        original_figure = children[0]['props']['children'][1]['props']['figure']

        image_bytes = pio.to_image(original_figure, format="png")
        image = Image.open(io.BytesIO(image_bytes))

        image_width, image_height = image.size
        left = (image_width - img_width) / 2
        top = (image_height - img_height) / 2
        right = (image_width + img_width) / 2
        bottom = (image_height + img_height) / 2

        cropped_image = image.crop((left, top, right, bottom))

        fig_cropped = go.Figure()
        fig_cropped.update_xaxes(
            visible=False,
            range=[0, img_width * scale_factor]
        )
        fig_cropped.update_yaxes(
            visible=False,
            range=[0, img_height * scale_factor],
            scaleanchor="x"
        )
        fig_cropped.add_layout_image(
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=cropped_image)
        )
        fig_cropped.update_layout(
            width=img_width * scale_factor,
            height=img_height * scale_factor,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )

        cropped_image_base64 = image_to_base64(cropped_image)

        return dbc.Card([
            html.H5('cropped image'),
            dcc.Graph(figure=fig_cropped,
                      config={'displayModeBar': False},  # Disable the mode bar
                      style={
                          'width': '100%',
                          'height': '100%',  # Take full height of the card
                          'padding': 0,  # Remove padding to fit the figure completely
                      }, ),
        ],
            style={
                'width': '100%',
                'height': '400px',
                'textAlign': 'center',
                'padding': '20px'
            },
        ), cropped_image_base64

    return no_update


@app.callback(
    Output("predicted-class", "children"),
    Input("predict-button", "n_clicks"),
    State("cropped-image-store", "data")  # Get the cropped image data from the Store
)
def predict_image(n_clicks, cropped_image_data):
    if n_clicks and cropped_image_data:
        # Convert the base64-encoded string back to a PIL Image object
        cropped_image = base64_to_image(cropped_image_data)

        # Call the external predict function to get the predicted class
        predicted_class = predict(cropped_image)  # Replace "predict" with your actual prediction function
        return f"Predicted Class: {predicted_class}"
    return no_update


@app.callback(
    Output("video-modal", "is_open"),
    Output("take-photo-btn", "n_clicks"),
    Input("open-modal-btn", "n_clicks"),
    Input("take-photo-btn", "n_clicks"),
    State("video-modal", "is_open"),
)
def toggle_modal(open_clicks, take_photo_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return False, 0
    prop_id = ctx.triggered[0]["prop_id"]
    if "open-modal-btn" in prop_id:
        return not is_open, 0
    elif "take-photo-btn" in prop_id and is_open:
        return not is_open, take_photo_clicks + 1
    raise PreventUpdate



if __name__ == '__main__':
    app.run_server(debug=True)
