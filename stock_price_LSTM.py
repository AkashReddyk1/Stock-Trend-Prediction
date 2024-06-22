import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

st.title("Stock Predictor using LSTM Web App")
ticker=st.text_input("Enter Stock Ticker:",'AAPL')

# Define the time period
start = '2014-01-01'
end = datetime.today().strftime('%Y-%m-%d')

# Fetch data
df = yf.download(ticker, start=start, end=end)
st.subheader("Data from 2014 to Today")
st.write(df.describe())

df=df.drop(['Adj Close','High','Low','Open','Volume'],axis=1)
df=df.reset_index()


st.subheader("Close Price vs Date")
fig=plt.figure(figsize=(12,6))
plt.plot(df.Date,df.Close)
st.pyplot(fig)

st.subheader("Close Price vs SMA 100")
sma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.Date,df.Close)
plt.plot(df.Date,sma100,'r')
st.pyplot(fig)

st.subheader("Close Price vs SMA 200 & SMA 100")
sma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.Date,df.Close,'b')
plt.plot(df.Date,sma100,'r')
plt.plot(df.Date,sma200,'g')
st.pyplot(fig)

dfc=df.drop(['Date'],axis=1)


# Split the data into 75% train and 25% test
data_train, data_test = train_test_split(dfc, test_size=0.25,random_state=0,shuffle=False)
print(data_train.shape)
print(data_test.shape)

scaler = MinMaxScaler(feature_range=(0, 1))

model=load_model('predict_price.h5')

past_100_days=data_train.tail(100)
#print(past_100_days)
final_test_df = pd.concat([past_100_days, data_test], ignore_index=True)

input_data=scaler.fit_transform(final_test_df)
print("Input_Data Shape",input_data.shape)

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
# print(x_test)
x_test,y_test=np.array(x_test),np.array(y_test)
print(x_test.shape)
print(y_test.shape)

y_predict=model.predict(x_test)
print(y_predict.shape)

y_predict=scaler.inverse_transform(y_predict)
y_test=scaler.inverse_transform(y_test.reshape(1,-1))

st.subheader("Original Price vs Predicted")
fig=plt.figure(figsize=(12,6))
split_index = int(len(df) * 0.75)
print("Split Index :",split_index)
plt.plot(df.Date[split_index:],dfc[split_index:],'b',label='Original Price')
plt.plot(df.Date[split_index:],y_predict,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Forecast next 30 days
st.subheader("Forecasted Price for next 30 days")
dfp = dfc.copy()
y_forcasted = []

for i in range(30):
    x_list = dfp.tail(100)
    x_next = scaler.transform(x_list)
    x_forc=[]
    x_forc.append(x_next)
    x_forc=np.array(x_forc)
    y_forc_sc = model.predict(x_forc)
    y_forc = scaler.inverse_transform(y_forc_sc)
    dfp.loc[len(dfp.index)] = y_forc[0][0]
    y_forcasted.append(y_forc[0][0])
print(y_forcasted)

fig=plt.figure(figsize=(12,6))
split_index = len(df)
# plt.plot(dfp[0:split_index],'b',label='Original Price')
plt.plot(range(1,31),dfp[split_index:],'r',label='Forcasted Price')
plt.xlabel('Upcoming days')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# print("y_predict", y_predict)
# print("y_test", y_test)
# st.subheader("Forcasted Price for next 30 days")
# dfp=dfc
# y_forcasted=[]
# for i in range(30) :
#     xlist=dfp.tail(100)
#     x_next=scaler.fit_transform(xlist)
#     print(x_next)
#     xforc=[]
#     xforc.append(x_next)
#     xforc=np.array(xforc)
#     print(xforc.shape)
#     yforc_sc=model.predict(xforc)
#     print(yforc_sc,"yforc_sc")
#     yforc=scaler.inverse_transform(yforc_sc)
#     print(yforc.shape)
#     dfp.loc[len(dfp.index)] = yforc[0][0]
#     y_forcasted.append(yforc)
# print(y_forcasted)
