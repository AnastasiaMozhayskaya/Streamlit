import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_option('deprecation.showPyplotGlobalUse', False)
# Задать стиль графиков.св
sns.set(style='whitegrid')

def data_separation(data):
    # Выбор колонок для деления на фичи и таргет
    st.sidebar.subheader("Выбор признаков для модели")
    features = st.sidebar.multiselect("Выберите признаки", options=data.columns, default=list(data.columns[:-1]))
    target = st.sidebar.selectbox("Выберите целевой признак", options=data.columns, index=len(data.columns)-1)
    data_features = data[features]
    data_target = data[target]
    X, X_test, y, y_test = train_test_split(data_features, data_target, random_state=42, test_size=0.25)

    return X, X_test, y, y_test


# Стадартизация данных и обучение модели
def train_model(X, X_test, y, y_test):
    scale = StandardScaler()
    X = scale.fit_transform(X)
    X_test = scale.transform(X_test)

    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    
    return model, accuracy

# Графики
def show_bar_chart(data):
    columns = data.columns[:-1]  # Названия столбцов
    weights = abs(data.iloc[-1, :-1]) # Веса столбцов

    st.markdown('### Оценка важности признаков модели LogisticRegression')
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=weights.sort_values(ascending = False), y=columns, palette='viridis', ax=ax)
    plt.xlabel('Вес признаков', labelpad=11, fontsize=13)
    plt.ylabel('Признаки', labelpad=11, fontsize=13)
    st.pyplot(fig)

def scatterplot(data):
    st.markdown('### Диаграмма рассеяния')
    st.sidebar.subheader("Диаграмма рассеяния")
    select_box1 = st.sidebar.selectbox(label='Ось X scatterplot', options=data.columns)
    select_box2 = st.sidebar.selectbox(label='Ось Y scatterplot', options=data.columns[::-1])
    select_box3 = st.sidebar.selectbox(label='Деление на группы по признаку', options=data.columns)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=data, x=select_box1, y=select_box2, hue=select_box3, ax=ax)
    st.pyplot(fig)

def barplot(data):
    st.markdown('### Столбчатая диаграмма')
    st.sidebar.subheader("Столбчатая диаграмма")
    select_box1 = st.sidebar.selectbox(label='Ось X1 barplot', options=data.columns)
    select_box2 = st.sidebar.selectbox(label='Ось Y1 barplot', options=data.columns[::-1])
    color_sequence = px.colors.qualitative.Plotly
    fig = px.bar(data_frame=data, x=select_box1, y=select_box2, color_discrete_sequence=color_sequence)
    fig.update_layout(width=9*80, height=7*80) 
    st.plotly_chart(fig)

def histogram(data):
    st.markdown('### Гистограмма')
    st.sidebar.subheader("Гистограмма")
    select_box = st.sidebar.selectbox(label='Выбор признака', options=data.columns)
    histogram_slider = st.sidebar.slider(label="Количество интервалов",min_value=5, max_value=100, value=30)
    fig = px.histogram(data_frame=data, x=select_box, nbins=histogram_slider, opacity=0.7, barmode='stack')
    fig.update_layout(width=9*80, height=7*80)
    fig.update_traces(marker=dict(line=dict(color='black', width=1)))
    st.plotly_chart(fig)

# Главная функция
def main():
    st.title('Модель Логистической регрессии')

    text = "#### Необходимо загрузить предобработанный файл с данными, он будет поделен на обучающую и тестовую выбороки. \
    Модель сама нормализует данные и рассчитает метрику качества *<span style='color:red'>Accuracy</span>*"

    st.write(text, unsafe_allow_html=True)

    # Загрузка файла CSV
    file = st.file_uploader('Загрузите файл с данными CSV', type='csv')
    
    if file is not None:
        data = pd.read_csv(file, index_col=0)

        st.sidebar.header('Панель инструментов :gear:')

        if st.sidebar.checkbox('Данные.'):
            st.write(data)
        
        # Обучение модели

        X, X_test, y, y_test = data_separation(data)
       # st.dataframe(X)

        model, accuracy = train_model(X, X_test, y, y_test)
        
        # Вывод результатов
        st.markdown(f"<h2 style='font-size: 20px; color: blue;'>Accuracy Логистической регрессии: \
                    <span style='color:red; font-size:30px'>{accuracy:.4f}</span></h2>", unsafe_allow_html=True)

        weights = np.insert(model.coef_, 0, model.intercept_)
        data.loc[len(data)] = weights
        show_bar_chart(data)
        scatterplot(data)
        histogram(data)
        barplot(data)
        

# Запуск приложения
if __name__ == '__main__':
    main()