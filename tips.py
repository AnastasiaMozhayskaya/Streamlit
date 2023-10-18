import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Код используется для настройки конфигурации страницы в приложении Streamlit
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

st.set_option('deprecation.showPyplotGlobalUse', False)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    """ Utility function for loading the autompg dataset as a dataframe."""
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')

    return df


# load dataset
data = load_data()
numeric_columns = data.select_dtypes(['float64', 'int64']).columns

st.sidebar.header('Панель инструментов')

# st.sidebar.subheader('Гистограмма')
# time_hist_color = st.sidebar.selectbox('Color by', ('temp_min', 'temp_max')) 

# st.sidebar.subheader('Donut chart parameter')
# donut_theta = st.sidebar.selectbox('Select data', ('q2', 'q3'))

"""
## Исследование по чаевым (датасет [tips.csv](https://github.com/mwaskom/seaborn-data/blob/master/tips.csv))

"""

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)

checkbox = st.sidebar.checkbox("Данные.")
if checkbox:
    # st.write(data)
    st.dataframe(data=data)

st.markdown('### Гистограмма total_bill (tip, size)')
st.sidebar.subheader("Гистограмма")
select_box3 = st.sidebar.selectbox(label="Признак", options=numeric_columns)
histogram_slider = st.sidebar.slider(label="Количество интервалов",min_value=5, max_value=100, value=30)
color = st.sidebar.selectbox('Цвет графика', ('cornflowerblue', 'silver', 'darkcyan'))
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(data[select_box3], bins=histogram_slider, ax=ax, kde=False, color=color)
st.pyplot(fig)


st.markdown('### Диаграмма рассеяния')
st.sidebar.subheader("Диаграмма рассеяния")
select_box1 = st.sidebar.selectbox(label='Ось X', options=data.columns)
select_box2 = st.sidebar.selectbox(label='Ось Y', options=data.columns[::-1])
select_box3 = st.sidebar.selectbox(label='Деление на группы по признаку', options=data.columns)
sns.relplot(x=select_box1, y=select_box2, data=data, hue=select_box3)
st.pyplot()

st.markdown('### График, связывающий total_bill, tip, и size')
sns.pairplot(data=data[['total_bill', 'tip', 'size']], height=2)
st.pyplot()

st.markdown('### График зависимости размера счета от дня недели')
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=data, x='day', y='total_bill')
st.pyplot(fig)

st.markdown('### Box plot c суммой всех счетов за каждый день')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="day", y="total_bill", hue="time", data=data)
st.pyplot(fig)

st.markdown('### Гистограммы чаевых на обед и ланч')
g = sns.FacetGrid(data, col="time")
g.map(sns.histplot, "tip")
st.pyplot(g)

st.markdown('### Диаграммы рассеяния для каждого пола с разбивкой на курящих/некурящих')
s = sns.FacetGrid(data=data, col="sex", hue='smoker')
s.map(sns.scatterplot, 'total_bill', 'tip')
s.add_legend(title='Smoker')
st.pyplot(s)