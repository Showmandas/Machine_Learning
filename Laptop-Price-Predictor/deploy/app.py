import streamlit as st
import pickle
import numpy as np
# from sklearn.compose import ColumnTransformer
#import the model
pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('datasets.pkl','rb'))
st.title('Laptop Price Predictor')


#brand Name
company=st.selectbox('Brand',df['Company'].unique())

#type of laptops
lap_types=st.selectbox('Type',df['TypeName'].unique())

#Ram
ram=st.selectbox('Ram(GB)',[2,4,8,12,24,32,64])

#weight
weight=st.number_input('Weight of laptop')

#Touchscreen

touchscreen=st.selectbox('Touchscreen',['No','Yes'])

#IPS
ips=st.selectbox('IPS',['No','Yes'])

#screen size
screen_size=st.number_input('Screen Size')

#resolution
resolution=st.selectbox('Screem Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#CPU
cpu=st.selectbox('CPU Brand',df['cpu_brand'].unique())
hdd=st.selectbox('HDD(GB)',[0,128,256,512,1024,2048])
ssd=st.selectbox('SSD(GB)',[0,8,128,256,512,1024])
gpu=st.selectbox('GPU Brand',df['Gpu_brands'].unique())
os=st.selectbox('Operating System',df['os'].unique())

if st.button('Predict Laptop Price'):
    # ppi = None
    if touchscreen == 'Yes':
        touchscreen=1
    else:
        touchscreen=0

    if ips == 'Yes':
        ips=1
    else:
        ips=0

    X_res=int(resolution.split('x')[0])
    Y_res=int(resolution.split('x')[1])
    ppi=((X_res ** 2)+(Y_res ** 2)) ** 0.5/screen_size
    query = np.array([company,lap_types,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query=query.reshape(1,12)
    st.title(int(np.exp(pipe.predict(query)[0])))

