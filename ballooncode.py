import streamlit as st
import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO
from huggingface_hub import HfFolder
import torch
from diffusers import DiffusionPipeline



# Streamlit UI
st.title("Balloon Vision")

if st.button('Click here to see what our balloon is seeing right now!'):
    with st.spinner('Loading! Wait time is around 2 minutes'):
        # Set your Hugging Face API key here (be cautious with your API key)
        hf_api_key = "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"

        # Save the API key in Hugging Face folder (this method avoids exposing the API key in your environment variables)
        HfFolder.save_token(hf_api_key)

        # Now you can directly use the API for accessing models, etc., without needing to log in manually through notebook_login
        model_id = "CompVis/stable-diffusion-v1-4"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pipe = DiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
        pipe = pipe.to(device)

        import requests
        import csv
        import io
        import requests
        import pandas as pd
        import io

        # Google Sheet ID and sheet name
        sheet_id = "1dFDXF94fu2UweY9r0cNijWwIjCjsemsYZ4foXTRFW14"
        sheet_name = "Sheet1"

        # URL for CSV format
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

        # Fetch the CSV data
        response = requests.get(url)
        response.raise_for_status()

        # Use pandas to read the CSV data into a DataFrame
        df = pd.read_csv(io.StringIO(response.text))

        # Assuming you want the last filled value from the first column (Python index starts at 0)
        column_values = df.iloc[:, 0]  # This selects all rows of the first column
        last_filled_prompt = column_values.dropna().iloc[-1]  # Drop NA values first, then select the last

        print(last_filled_prompt)

        import os

        # Set your prompt
        prompt = last_filled_prompt

        # Adjust parameters for faster generation
        gen_kwargs = {
            "num_inference_steps": 10,  # Default is higher, reducing it can speed up the process
        }

        # Generate image with adjusted parameters
        image = pipe(prompt, **gen_kwargs).images[0]

        # Function to find the lowest unused index for the filename
        def find_lowest_unused_index(prefix, suffix):
            index = 0
            while True:
                filename = f"{prefix}{index}{suffix}"
                if not os.path.exists(filename):
                    return index
                index += 1

        # Find the lowest unused index
        index = find_lowest_unused_index("flightimage", ".png")

        # Save the image with the found index

        st.image(image, caption="View from our balloon")
        # Display the generated image
        image.show()
