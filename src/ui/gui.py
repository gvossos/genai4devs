import gradio as gr
import fitz
from PIL import Image

from src.services.service import kickoff_workflow, MicroService


# Render the pdf
def render_file(file):
    doc = fitz.open(file.name)
    page = doc[0]
    #Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image

   
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Fund Analyzer AI Agent")
        
        # Input for the JSON list of investments
        transaction_input = gr.TextArea(label="Enter Elektra monthly switches using JSON:", 
                                        value='''{"transactions":[{"SELL":{"MStarID":"F0AUS05EML","APIR":"IML0001AU","Benchmark":"XIUSA04GAR","Investment":"Investors Mutual WS Aus Smaller Co"},"BUY":{"MStarID":"F0000101OM","APIR":"BFL3779AU","Benchmark":"XIUSA04GAR","Investment":"Bennelong Emerging Companies Fund"}}]}''',
                                        lines=6)
        
        asset_allocation_views_uploader_button = gr.UploadButton("Upload Quarterly Asset Allocation View PDF", file_types=[".pdf"])
        
        watchlist_uploader_button = gr.UploadButton("Upload Weekly Monitoring watch-list PDF", file_types=[".pdf"])
        
        workflow_button = gr.Button("Kick-off workflow process...")
        
        status_state = gr.State(value="Waiting...")
        result_output = gr.Textbox(label="Result", interactive=False, lines=10)
        

        show_asset_allocation_view_img = gr.Image(label='Upload Asset Allocation View PDF', height=340)
        
        show_watchlist_view_img = gr.Image(label='Upload Weekly Monitoring watch-list', height=340)
        
        # When the button is clicked, call workflow and update status and result
        workflow_button.click(
            fn=kickoff_workflow,  # This function needs to be adapted to handle investment JSON input and generate URLs dynamically
            inputs=[
                transaction_input,
                asset_allocation_views_uploader_button, 
                watchlist_uploader_button,
                status_state
            ],
            outputs=[status_state, result_output]
        )
        
        # Event handler for uploading a PDF
        asset_allocation_views_uploader_button.upload(
            fn=render_file, 
            inputs=[asset_allocation_views_uploader_button],
            outputs=[show_asset_allocation_view_img]
        )
        
         # Event handler for uploading a PDF
        watchlist_uploader_button.upload(
            fn=render_file, 
            inputs=[watchlist_uploader_button],
            outputs=[show_watchlist_view_img]
        )
        
        
    return demo


