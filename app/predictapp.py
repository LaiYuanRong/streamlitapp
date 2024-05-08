import pickle
import streamlit as st

from pyecharts.charts import Bar, Grid, Gauge
from pyecharts import options as opts
from streamlit_echarts import st_pyecharts
import os

path = os.getcwd().replace("\\", "/")

import pandas as pd

s = '''
    border: none !important; 
    -webkit-box-shadow: 5px 5px 5px rgba(0, 0, 0, .2), -5px -5px 5px #fff;
    box-shadow: 5px 5px 5px rgba(0, 0, 0, .2), -5px -5px 5px #fff;
    border-radius: 0.75rem !important;
    text-align: center;
    padding: 1em;
    background: #ff4b4b;
    border-radius: 0.75rem;
'''

st.markdown('''
    <style>
        iframe {
                	border: none !important;
                	-webkit-box-shadow: 5px 5px 5px rgba(0, 0, 0, .2), -5px -5px 5px #fff;
                	box-shadow: 5px 5px 5px rgba(0, 0, 0, .2), -5px -5px 5px #fff;
                	border-radius: 0.75rem !important;
                	text-align: center;
                	padding: 1em;
                    background: rgba(245,245,245,1);
                }
    </style>
    ''', unsafe_allow_html=True)
    
def gauge(title, v):
    g = (
        Gauge()
        .add("", [(title, v)])
        .set_global_opts(
            title_opts=opts.TitleOpts(title="预测结果", pos_left="center", pos_top="25px"),
            legend_opts=opts.LegendOpts(is_show=False)
        )
        .set_series_opts(
            itemstyle_opts=opts.ItemStyleOpts(color="rgba(255,75,75,0.6)")
        )
    )

    st_pyecharts(g, height=460)

with open(path+'/逻辑回归模型.pkl', 'rb') as f:
    model = pickle.load(f)

col = ['Age', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Vintage']
col1 = ['年龄', '之前是否投保', '驾龄', '车辆是否发生过损坏', '年度保费', '客户与公司建立联系的时长']
d = {"是":1, "否":0}

data = {i:None for i in col}

st.sidebar.markdown(f'<h1 id="title" style="text-align: center; font-size: 20px; margin-bottom: 1rem; color: white; {s}">预测参数</h1>', unsafe_allow_html=True)
with st.sidebar.form("模型参数"):
    for i, j in zip(col, col1):
        if i==col[1]:
            data[i] = d[st.selectbox(j, list(d.keys()), index=0)]
        elif i==col[3]:
            data[i] = d[st.selectbox(j, list(d.keys()), index=0)]
        else:
            data[i] = st.number_input(j, key=i, step=1)
        
    submitted = st.form_submit_button("开始预测", use_container_width=True)

st.markdown(f'<h1 id="title" style="text-align: center; font-size: 20px; margin-bottom: 1rem; color: white; {s}">逻辑回归模型预测结果</h1>', unsafe_allow_html=True)

V = {1:"感兴趣", 0:"不感兴趣"}

if submitted:
    v = V[model.predict(pd.DataFrame([data]))[0]]
    p = round(100-model.predict_proba(pd.DataFrame([data])).flatten()[0]*100, 2)
    
    gauge("预测结果:"+v, p)
    
else:
    gauge("当前未进行预测", 50.00)

t = '''              precision    recall  f1-score   support

           0       1.00      0.88      0.93    114319
           1       0.00      0.21      0.00        14

    accuracy                           0.88    114333
   macro avg       0.50      0.55      0.47    114333
weighted avg       1.00      0.88      0.93    114333'''

with st.expander("模型特征重要性与评估", True):
    st.image("特征重要性.png", use_column_width=True)
    st.write(t.splitlines())
    
