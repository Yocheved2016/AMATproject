import dash_bootstrap_components as dbc
from dash import dcc, html

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

layout = dbc.Container([
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
