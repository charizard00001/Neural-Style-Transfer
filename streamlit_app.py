import streamlit as st
from PIL import Image
from nst_model import run_style_transfer, tensor_to_image

def display_nst_image(content_img_path, style_img_path):
    output_img, _ = run_style_transfer(content_img_path, style_img_path)
    output_image = tensor_to_image(output_img)
    st.image(output_image, caption='Stylized Image', use_column_width=True)

st.title("Neural Style Transfer App")

st.write("Upload a content image and a style image, then click the button to apply neural style transfer.")

content_img = st.file_uploader("Choose a content image", type=['png', 'jpg', 'jpeg'], key="content")

style_img = st.file_uploader("Choose a style image", type=['png', 'jpg', 'jpeg'], key="style")

if content_img and style_img:
    st.image(content_img, caption='Content Image', use_column_width=True)
    st.image(style_img, caption='Style Image', use_column_width=True)
    
    content_img_path = "temp_content_img.jpg"
    style_img_path = "temp_style_img.jpg"
    with open(content_img_path, "wb") as f:
        f.write(content_img.getbuffer())
    with open(style_img_path, "wb") as f:
        f.write(style_img.getbuffer())
 
    if st.button("Generate Stylized Image"):
        with st.spinner("Generating stylized image..."):
            try:
                display_nst_image(content_img_path, style_img_path)
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.warning("Please upload both a content image and a style image.")
