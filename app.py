import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title='Student Pass Predictor', page_icon='🎓')

@st.cache_resource
def build_model():
    """สร้างข้อมูลชุดเดิม (seed เดียวกัน) แล้วเทรนโมเดลเองตอนเปิดเว็บ"""
    np.random.seed(42)
    n = 500
    study       = np.round(np.random.uniform(0, 8, n), 1)
    sleep       = np.round(np.random.uniform(4, 10, n), 1)
    absences    = np.random.poisson(4, n)
    part_time   = np.random.choice(['No', 'Yes'], n, p=[0.7, 0.3])
    quiz        = np.clip(30 + 5*study + np.random.normal(0, 10, n), 0, 100).round(1)
    assignments = np.clip(np.random.poisson(7, n), 0, 10)

    z = (0.8*study + 0.4*(sleep-6) - 0.25*absences
         + 0.06*(quiz-50) + 0.3*(assignments-5)
         - 0.6*(part_time=='Yes') - 2.2)
    passed = (np.random.rand(n) < 1/(1+np.exp(-z))).astype(int)

    df = pd.DataFrame({'Study_Hours': study, 'Sleep_Hours': sleep, 'Absences': absences,
                       'Quiz_Score': quiz, 'Assignments': assignments,
                       'Part_Time_Job': part_time, 'Passed': passed})
    for col in ['Sleep_Hours', 'Quiz_Score']:
        df.loc[np.random.rand(n) < 0.04, col] = np.nan
    df = pd.concat([df, df.sample(10, random_state=7)], ignore_index=True)
    df.drop_duplicates(inplace=True)

    X = df.drop('Passed', axis=1)
    y = df['Passed']
    num_cols = ['Study_Hours', 'Sleep_Hours', 'Absences', 'Quiz_Score', 'Assignments']
    cat_cols = ['Part_Time_Job']
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('sc',  StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder())]), cat_cols),
    ])
    pipe = Pipeline([('pre', pre),
                     ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
    pipe.fit(X, y)
    return pipe

model = build_model()

st.title('🎓 ระบบทำนายโอกาสสอบผ่านของนักศึกษา')
st.caption('มินิโปรเจค: Student Performance Prediction')

with st.sidebar:
    st.header('📝 กรอกข้อมูลการเรียน')
    study  = st.slider('ชั่วโมงอ่านหนังสือ/วัน', 0.0, 8.0, 3.0, 0.5)
    sleep  = st.slider('ชั่วโมงนอน/วัน', 4.0, 10.0, 7.0, 0.5)
    absen  = st.slider('จำนวนครั้งการขาดเรียน', 0, 15, 3)
    quiz   = st.slider('คะแนนควิซเฉลี่ย', 0, 100, 60)
    assign = st.slider('จำนวนงานที่ส่ง (จาก 10)', 0, 10, 7)
    job    = st.selectbox('ทำงานพาร์ทไทม์', ['No', 'Yes'])

X_in = pd.DataFrame([[study, sleep, absen, quiz, assign, job]],
                    columns=['Study_Hours', 'Sleep_Hours', 'Absences',
                             'Quiz_Score', 'Assignments', 'Part_Time_Job'])

if st.button('🔮 ทำนายผล', type='primary'):
    prob = float(model.predict_proba(X_in)[0][1])
    st.metric('โอกาสสอบผ่าน', f'{prob*100:.1f}%')
    st.progress(prob)
    if prob >= 0.7:
        st.success('✅ แนวโน้มดี รักษาวิถีนี้ไว้!')
    elif prob >= 0.4:
        st.warning('⚠️ เริ่มเสี่ยง ลองเพิ่มชั่วโมงทบทวนบทเรียนอีกนิด')
    else:
        st.error('🚨 เสี่ยงสูง! ควรปรึกษาอาจารย์และปรับแผนการเรียนด่วน')

if os.path.exists('results.png'):
    st.markdown('---')
    st.subheader('📊 ประสิทธิภาพโมเดล')
    st.image('results.png', use_column_width=True)