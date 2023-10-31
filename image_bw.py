import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from skimage import io
from sklearn.decomposition import TruncatedSVD


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide", page_title="Image SVD")

st.write("## Ощути прелесть SVD в Изображениях")
st.write(
    "Попробуйте загрузить изображение, чтобы ощутить всю магию SVD в Изображениях :smile:"
)
st.sidebar.write("### Загрузите черно-белое изображение :gear:")

MAX_FILE_SIZE = 5 * 2048 * 2048  # 10MB


def k_function(image):
    image = io.imread(image, as_gray=True)
    U, sing_values, V = np.linalg.svd(image)
    sigma = np.zeros(shape=image.shape)
    np.fill_diagonal(sigma, sing_values)
    st.sidebar.subheader("Выбрать кол-во сингулярных чисел")
    top_k = st.sidebar.slider(label="", min_value=0, max_value=500, value=30) 
    top_k = top_k
    modified_U = U[:, :top_k]
    modified_sigma = sigma[:top_k, :top_k]
    modified_V = V[:top_k, :]
    modified_image = modified_U @ modified_sigma @ modified_V
    return modified_image

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = io.imread(upload, as_gray=True)
    col1.write("Исходное изображение :camera:")
    plt.imshow(image, cmap='gray')
    plt.grid(False)
    col1.pyplot()

    fixed = k_function(upload)
    col2.write("Измененное изображение :wrench:")
    plt.imshow(fixed, cmap='gray')
    plt.grid(False)
    col2.pyplot()


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("Загруженный файл слишком велик. Пожалуйста, загрузите изображение размером менее 10 МБ.")
    else:
        fix_image(upload=my_upload)
else:
    try:
        fix_image("https://bogatyr.club/uploads/posts/2023-03/1679122728_bogatyr-club-p-volk-grafika-foni-oboi-43.jpg")
    except:
        fix_image("volk.jpg")