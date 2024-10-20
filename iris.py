import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 제목과 설명

st.title("Iris Species Predictor")
st.write("""
         dataset
         """)

iris = load_iris()
X = iris.data # 0:setosa 1:vesicolor 2:virginica
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]

#사이드바에서 입력 받기

def user_input_features():
    # 사이드바 헤더
    st.sidebar.header("Input Parameters")

    # 사용자의 입력을 받기 위한 슬라이더
    sepal_length = st.sidebar.slider("Sepal length (cm)",
                                      float(df['sepal_length'].min()),
                                      float(df['sepal_length'].max()))
    sepal_width = st.sidebar.slider("Sepal width (cm)",
                                     float(df['sepal_width'].min()),
                                     float(df['sepal_width'].max()))
    petal_length = st.sidebar.slider("Petal length (cm)",
                                      float(df['petal_length'].min()),
                                      float(df['petal_length'].max()))
    petal_width = st.sidebar.slider("Petal width (cm)",
                                     float(df['petal_width'].min()),
                                     float(df['petal_width'].max()))

    # 입력된 값들을 데이터로 정의
    data = {
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    }

    # DataFrame 생성
    features = pd.DataFrame(data, index=[0])  # index=[0] 유지
    return features

input_df = user_input_features()
# print(input(df)

# 사용자 입력 값 표시
st.subheader("User Input Parameters")
st.write(input_df)

# RandomForestClassifier 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 예측
prediction = model.predict(input_df.to_numpy())
# print(prediction, iris.target_names)
prediction_proba = model.predict_proba(input_df.to_numpy())
st.subheader("Prediction Probability")
st.write(prediction_proba)
#히스토그램
st.subheader("Histogram of Features")

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax = ax.flatten()
for i, column in enumerate(df.columns):
    sns.histplot(df[column], kde=True, ax=ax[i])
    ax[i].set_title(f'Plot of {df.feature_names[i]}')
plt.tight_layout()

st.pyplot(fig)

st.subheader("Correlation Matrix")
numerical_df = df.drop('species', axis=1)
corr = numerical_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', lineWidths=0.5)
plt.tight_layout()
st.pyplot(plt)

# 페어플롯

st.subheader('Pairplot')
fig = sns.pairplot(df, hue="species").fig
plt.tight_layout()
st.pypplot(fig)

st.subheader('Feature Importance')
importtances = model.feature_importances_

indices = np.argsort(importtances)[::-1]

plt.figure(figsize= (10,4))
plt.title("Feature Importance")
plt.bar(range(X.shpae[1]), importtances[indices], align = "center")
plt.xticks(range(X.shape[1]), [iris.feature_names[i] for i in indices])
plt.xilim([-1, X.shape[1]])
st.pyplot(plt)
