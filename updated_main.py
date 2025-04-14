import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from updated_util import classify, set_background

def show_output_chart(predictions, class_names):
    colors = ['#fddccf', '#fce4ec', '#8e44ad', '#d6eaf8', '#fddccf']  # customize as needed

    fig = go.Figure()

    for i, (score, label) in enumerate(zip(predictions, class_names)):
        fig.add_trace(go.Bar(
            x=[score * 100],
            y=[label],
            orientation='h',
            marker=dict(color=colors[i % len(colors)]),
            text=f'{int(score * 100)}%',
            textposition='inside',
            insidetextanchor='start',
            hoverinfo='none'
        ))

    fig.update_layout(
        title='Output',
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
        yaxis=dict(autorange='reversed'),
        height=300,
        margin=dict(l=100, r=20, t=40, b=20),
        plot_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)


set_background("BackGround.jpg")

with st.container():
    st.markdown("<div class='block-container'>", unsafe_allow_html=True)

# set title
st.title('Multiple Lung Disease classification')

# set header
st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

from tensorflow.keras.layers import DepthwiseConv2D

# Patch to remove 'groups' argument for compatibility
original_init = DepthwiseConv2D.__init__

def patched_init(self, *args, **kwargs):
    kwargs.pop('groups', None)  # Ignore 'groups' if present
    original_init(self, *args, **kwargs)

DepthwiseConv2D.__init__ = patched_init


# load classifier
model = load_model("converted_keras/keras_model.h5")

# load class names
with open("converted_keras/labels.txt", 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    st.subheader("Output")

    # write classification
    class_name, conf_score, all_preds = classify(image, model, class_names)
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
    
    # show predicted label
    st.write("### Predicted: `{}` with {:.2f}% confidence".format(class_name, conf_score * 100))

    # display horizontal bar chart like your image
    
    for i, score in enumerate(all_preds):
        st.markdown(f"**{class_names[i]}**")
        st.progress(float(score))
    st.markdown("</div>", unsafe_allow_html=True)
