from dash import Output, Input, State, no_update, callback_context
from GUI.utils import VideoCamera, image_to_base64, base64_to_image, predict
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from GUI.layout import img_width, img_height, scale_factor, parse_contents
import plotly.io as pio
from PIL import Image
import base64
import io
import dash_bootstrap_components as dbc
from dash import dcc, html

def register_callbacks(app):
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
            left = 0
            top = 0
            right = image_width
            bottom = image_height

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
                    source=cropped_image)
            )
            fig_cropped.update_layout(
                width=img_width * scale_factor,
                height=img_height * scale_factor,
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
            )

            cropped_image_base64 = image_to_base64(cropped_image)

            return parse_contents(fig_cropped, 'cropped image' ),cropped_image_base64

        return no_update

    @app.callback(
        Output("predicted-class", "children"),
        Input("predict-button", "n_clicks"),
        State("cropped-image-store", "data")
    )
    def predict_image(n_clicks, cropped_image_data):
        if n_clicks and cropped_image_data:
            cropped_image = base64_to_image(cropped_image_data)
            predicted_class = predict(cropped_image)
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