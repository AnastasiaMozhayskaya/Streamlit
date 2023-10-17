import yfinance as yf
import pandas as pd
import streamlit as st

"""
## Приложение для просмотра катеровок по компании Apple
#### О компании Apple
Apple — крупнейшая американская технологическая компания, которая производит электронику, программное обеспечение и цифровой контент. 
Первая компания в США, капитализация которой в 2018 году превысила 1 трлн долларов.
Apple — мировой лидер в области информационных технологий по выручке и третий по величине производитель мобильных телефонов после Samsung и Huawei. 
В 2023 году количество активных устройств компании превысило 2 млрд штук. Компания старается диверсифицировать бизнес, 
выступает новатором отрасли: планирует разработку собственных чипов беспроводной связи, создание электрокаров к 2025 году и 
активное развитие AR-технологий.
"""

st.title("Apple Inc (AAPL)")

ticker = 'AAPL'

data = yf.Ticker(ticker)

df = data.history(period='1y', start='1998-01-01', end='2023-10-15')

st.subheader('График цены закрытия')
st.line_chart(df.Close)
