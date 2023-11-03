import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Задать стиль графиков.
sns.set(style='darkgrid')
      
# Стадартизация данных и обучение модели
def train_model(data, data1):
    X = data.iloc[:, :-1]  # Признаки
    y = data.iloc[:, -1]  # Целевая переменная
    X_test = data1.iloc[:, :-1]  # Признаки
    y_test = data1.iloc[:, -1]  # Целевая переменная

    scale = StandardScaler()
    X = scale.fit_transform(X)
    X_test = scale.transform(X_test)

    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred==y_test)
    
    return model, accuracy

# Графики
def show_bar_chart(data):
    columns = data.columns[:-1]  # Названия столбцов
    weights = abs(data.iloc[-1, :-1]) # Веса столбцов

    fig, ax = plt.subplots()
    sns.barplot(x=weights.sort_values(ascending = False), y=columns, palette='viridis')
    plt.title('Оценка важности признаков модели LogisticRegression', fontsize=15)
    plt.xlabel('Вес признаков', labelpad=11, fontsize=13)
    plt.ylabel('Признаки', labelpad=11, fontsize=13)
    st.pyplot(fig)

def scatterplot(data):
    st.markdown('### Диаграмма рассеяния')
    st.sidebar.subheader("Диаграмма рассеяния")
    select_box1 = st.sidebar.selectbox(label='Ось X scatterplot', options=data.columns)
    select_box2 = st.sidebar.selectbox(label='Ось Y scatterplot', options=data.columns[::-1])
    fig = px.scatter(data, x=select_box1, y=select_box2)
    fig.update_layout(width=9*80, height=7*80)  
    st.plotly_chart(fig)

def barplot(data):
    st.markdown('### Столбчатая диаграмма')
    st.sidebar.subheader("Столбчатая диаграмма")
    select_box1 = st.sidebar.selectbox(label='Ось X1 barplot', options=data.columns)
    select_box2 = st.sidebar.selectbox(label='Ось Y1 barplot', options=data.columns[::-1])
    color_sequence = px.colors.qualitative.Plotly
    fig = px.bar(data_frame=data, x=select_box1, y=select_box2, color_discrete_sequence=color_sequence)
    fig.update_layout(width=9*80, height=7*80) 
    st.plotly_chart(fig)

def line(data):
    st.markdown('### Линейная диаграмма')
    st.sidebar.subheader("Линейная диаграмма")
    select_box1 = st.sidebar.selectbox(label='Ось X1 line', options=data.columns)
    select_box2 = st.sidebar.selectbox(label='Ось Y1 line', options=data.columns[::-1])
    color_sequence = px.colors.qualitative.Pastel
    fig = px.line(data_frame=data, x=select_box1, y=select_box2, color_discrete_sequence=color_sequence)
    fig.update_layout(width=9*80, height=7*80) 
    st.plotly_chart(fig)


# Главная функция
def main():
    st.title('Модель Логистической регрессии')

    text = "#### Необходимо загрузить предобработанные файлы обучающей и тестовой выборок при условии, что целевой признак является последним столбцом в данных. Модель сама нормализует данные и рассчитает метрику качества *<span style='color:red'>Accuracy</span>*."

    st.write(text, unsafe_allow_html=True)

    # Загрузка файла CSV
    file = st.file_uploader('Загрузите файл CSV для обучающей выборки', type='csv')

    # Загрузка файла CSV
    file1 = st.file_uploader('Загрузите файл CSV для тестовой выборки', type='csv')
    
    if file is not None:
        data = pd.read_csv(file, index_col=0)


    if file1 is not None:
        data1 = pd.read_csv(file1, index_col=0)

        st.sidebar.header('Панель инструментов :gear:')

        if st.sidebar.checkbox('Данные.'):
            st.write(data)
        
        # Обучение модели
        model, accuracy = train_model(data, data1)
        
        # Вывод результатов
        st.markdown(f"<h2 style='font-size: 20px; color: blue;'>Accuracy Логистической регрессии: \
                    <span style='color:red; font-size:30px'>{accuracy}</span></h2>", unsafe_allow_html=True)

        weights = np.insert(model.coef_, 0, model.intercept_)
        data.loc[len(data)] = weights
        show_bar_chart(data)
        scatterplot(data)
        barplot(data)
        line(data)

# Запуск приложения
if __name__ == '__main__':
    main()