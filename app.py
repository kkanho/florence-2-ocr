import os
import ast
import random
import copy
import torch
import streamlit as st
import pandas as pd
import numpy as np
from unittest.mock import patch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
from PIL import Image, ImageDraw

uploaded_image_path = "./uploaded_images"
output_path = "./output"

if not os.path.exists(uploaded_image_path):
    os.makedirs(uploaded_image_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

st.set_page_config(layout="wide")

def get_device_type():
    if torch.cuda.is_available():  
        return "cuda"
    else: 
        if (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            return "mps"
        else:
            return "cpu"
        
def fixed_get_imports(filename):
    # flash_attention
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

def run_model(task_prompt, image, model_id):

    device_type = get_device_type()
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True).to(device_type)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, clean_up_tokenization_spaces=False)


    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device_type)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def draw_ocr_bboxes(image, prediction):

    colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=2, outline=color)
        draw.text((new_box[0]+18, new_box[1]+12),
                    "{}".format(label),
                    align="right",
                    fill=color)
    return image

def ocr_with_region(image, model_id):
    task_prompt = '<OCR_WITH_REGION>'
    text_results = run_model(task_prompt, image, model_id)
    output_image = copy.deepcopy(image)
    output_image = draw_ocr_bboxes(output_image, text_results[task_prompt])

    return text_results, output_image


# Function to read and process the corresponding text file
def get_labels_from_json(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            raw_content = file.read()
            data = ast.literal_eval(raw_content)
            labels = data.get('<OCR_WITH_REGION>', {}).get('labels', [])
            return labels
    except Exception as e:
        return [str(e)]

def get_model_used_from_json(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            raw_content = file.read()
            data = ast.literal_eval(raw_content)
            model_used = data.get('<OCR_WITH_REGION>', {}).get('model', [])
            return str(model_used)
    except Exception as e:
        return [str(e)]

####################################################
#init
image = None
results = None

st.title("Florence 2 OCR")
col1, col2 = st.columns(2)

with st.sidebar:
    uploaded_img = st.file_uploader("Choose a image", type=['png', 'jpg'])
    model_id = st.selectbox("Choose a model", options=['microsoft/Florence-2-large', 'microsoft/Florence-2-large-ft' , 'microsoft/Florence-2-base', 'microsoft/Florence-2-base-ft'])


with col1:
    st.caption("Input")

    if uploaded_img is not None:
        image = Image.open(uploaded_img)

        st.image(image)
        st.text(uploaded_img.name)

with col2:
    st.caption("Output")
    if image is not None:
        results, output_image = ocr_with_region(image, model_id)
        st.image(output_image)
        
        filename, extension = uploaded_img.name.split('.', 1)

        # save the image
        model_used = model_id.split("/")[1]
        image.save(f"{uploaded_image_path}/{model_used}_{filename}.{extension}")
        output_image.save(f"{output_path}/output_{model_used}_{filename}.{extension}")

        results["<OCR_WITH_REGION>"]["model"] = model_id

        # save the output text
        with open(f"{output_path}/output_{model_used}_{filename}.json", "w") as text_file:
            text_file.write(str(results))



st.markdown("---")
st.caption("results text")
if results is not None:
    df = pd.DataFrame(results['<OCR_WITH_REGION>'], columns=['labels', 'quad_boxes'])
    st.dataframe(df)
    st.json(results)

st.markdown("---")

# Get the list of image
image_files = [f for f in os.listdir(output_path) if f.endswith('.png')]
for image_file in image_files:
    st.markdown(f"<h6 style='text-align: center;'>{image_file}</h6>", unsafe_allow_html=True)
    cols = st.columns(3)

    original_image_path = os.path.join(uploaded_image_path, image_file.split('output_')[1])
    image_path = os.path.join(output_path, image_file)
    json_file_path = image_path.replace('.png', '.json')

    if os.path.exists(original_image_path): # Image file exists

        original_image = Image.open(original_image_path)
        cols[0].image(original_image, caption=f"", use_column_width=True)

        if os.path.exists(image_path):

            image = Image.open(image_path)
            cols[1].image(image, caption=f"", use_column_width=True)

        else:
            cols[1].text(f"{image_path} not found")

        if os.path.exists(json_file_path): # Text filet exists
            labels = get_labels_from_json(json_file_path)
            df = pd.DataFrame(labels, columns=["labels"])
            cols[2].dataframe(df, hide_index=True, use_container_width=True)

            model_used = get_model_used_from_json(json_file_path)
            st.caption(model_used)
        else:
            cols[2].text(f"{json_file_path} not found")
    else:
        cols[1].text(f"{original_image_path} not found")

