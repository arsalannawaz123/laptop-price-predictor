import pickle
import streamlit as st
import pandas as pd
import numpy as np

st.title("laptop Price predictor")
pipe = pickle.load(open('pipe.pkl', 'rb'))
# st.write(pipe.named_steps['step2'])

df= pickle.load(open('df.pkl', 'rb'))

# brand of the laptop
company = st.selectbox('Brand',df['Company'].unique())

# type of the laptop
type= st.selectbox('Type',df['TypeName'].unique())

# RAm of the laptop
ram = st.selectbox('Ram in GB',[2,4,6,8,12,16,32,64])

# CPU of the laptop
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

# weitght
weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1)

#Touch Screen
touch_screen= st.selectbox ('Touch Screen',['Yes','No'])

#IPS display

ips = st.selectbox('IPS Display', ['No', 'Yes'])

#Screen size
screen_size = st.number_input('Screen size (inches)', min_value=10.0, max_value=18.0, value=15.6, step=0.1)

# resoution
resolution = st.selectbox('Screen resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
# HDD
HDD = st.selectbox('HDD in GB', [0,128,256,512,1024,2048])
# SDD
SSD = st.selectbox('SSD in GB', [0,128,256,512,1024,2048])

#GPU
gpu = st.selectbox('GPU', sorted(df['Gpu brand'].unique()))
#OS
os = st.selectbox('Operating system', sorted(df['os'].unique()))


# now predict the price 

if st.button("Predict the price"):
    touch_screen= 1 if touch_screen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0 

    # ppi = pixels per inch, calculated from resolution and screen size
    x_res, y_res = map(int, resolution.split('x'))
    ppi = ((x_res ** 2) + (y_res ** 2)) ** 0.5 / screen_size


        # Same columns, same order as X_train
    query = pd.DataFrame([[company, type, ram, weight, touch_screen, ips,
                           ppi, cpu, HDD, SSD, gpu, os]],
                         columns=['Company', 'TypeName', 'Ram', 'Weight',
                                  'Touchscreen', 'IPS', 'ppi', 'Cpu brand',
                                  'HDD', 'SSD', 'Gpu brand', 'os'])

    # Model predicts log(price), so convert back with np.exp
   # Model predicts log(price) in INR, so convert back with np.exp
    price_inr = np.exp(pipe.predict(query)[0])

    # Exchange rates (update these when they change)
    INR_TO_PKR = 2.9
    PKR_PER_USD = 277
    MARKET_ADJUSTMENT = 1.0     # keep 1.0 for no adjustment, or use a factor like 1.5

    price_pkr = price_inr * INR_TO_PKR * MARKET_ADJUSTMENT
    price_usd = price_pkr / PKR_PER_USD

    st.header(f'Predicted price: Rs {int(price_pkr):,} PKR')
    st.subheader(f'About ${int(price_usd):,} USD')
    st.caption(f'(Model price in INR: {int(price_inr):,})')
        









