import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title='Student Pass Predictor', page_icon='🎓', layout='wide')
sns.set_theme(style='whitegrid')
NUM = ['Study_Hours','Sleep_Hours','Absences','Attendance_Pct','Quiz_Score','Assignments','GPA_Prev']
CAT = ['Part_Time_Job']

@st.cache_resource
def build_all():
    np.random.seed(42); n = 1500
    study = np.round(np.random.uniform(0, 8, n), 1)
    sleep = np.round(np.random.uniform(4, 10, n), 1)
    absences = np.random.poisson(4, n)
    attendance = np.clip(100 - absences*4 + np.random.normal(0, 5, n), 40, 100).round(1)
    gpa = np.clip(np.random.normal(2.5, 0.7, n), 0, 4).round(2)
    part_time = np.random.choice(['No', 'Yes'], n, p=[0.7, 0.3])
    quiz = np.clip(25 + 5*study + 2*(gpa-2.5) + np.random.normal(0, 8, n), 0, 100).round(1)
    assignments = np.clip(np.random.poisson(7, n), 0, 10)
    z = (0.7*study + 0.3*(sleep-6) - 0.2*absences + 0.05*(quiz-50)
         + 0.25*(assignments-5) + 0.5*(gpa-2.5) - 0.5*(part_time=='Yes') - 2.0)
    passed = (np.random.rand(n) < 1/(1+np.exp(-z))).astype(int)
    df = pd.DataFrame({'Study_Hours': study, 'Sleep_Hours': sleep, 'Absences': absences,
                       'Attendance_Pct': attendance, 'Quiz_Score': quiz, 'Assignments': assignments,
                       'GPA_Prev': gpa, 'Part_Time_Job': part_time, 'Passed': passed})
    for col in ['Sleep_Hours', 'Quiz_Score', 'Attendance_Pct']:
        df.loc[np.random.rand(n) < 0.03, col] = np.nan
    df = pd.concat([df, df.sample(15, random_state=7)], ignore_index=True)
    df.drop_duplicates(inplace=True)

    X = df.drop('Passed', axis=1); y = df['Passed']
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), NUM),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder())]), CAT)])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    models = {'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
              'Decision Tree': DecisionTreeClassifier(random_state=42),
              'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)}
    res, pipes = {}, {}
    for name, m in models.items():
        pipe = Pipeline([('pre', pre), ('model', m)])
        pipe.fit(X_tr, y_tr)
        yp = pipe.predict(X_te)
        res[name] = [accuracy_score(y_te, yp), precision_score(y_te, yp), recall_score(y_te, yp), f1_score(y_te, yp)]
        pipes[name] = pipe
    res_df = pd.DataFrame(res).T
    res_df.columns = ['Accuracy','Precision','Recall','F1-Score']
    best = res_df['F1-Score'].idxmax()
    cm = confusion_matrix(y_te, pipes[best].predict(X_te))
    rf = pipes['Random Forest']
    names = list(NUM) + list(rf.named_steps['pre'].named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(CAT))
    fi = pd.DataFrame({'feature': names, 'importance': rf.named_steps['model'].feature_importances_}).sort_values('importance', ascending=False)
    return df, pipes[best], best, res_df, cm, fi

df, model, best_name, res_df, cm, fi = build_all()

st.title('🎓 ระบบทำนายโอกาสสอบผ่านของนักศึกษา')
st.caption('มินิโปรเจค | Student Performance Prediction | ข้อมูล 1,500 รายการ')

m1, m2, m3, m4 = st.columns(4)
m1.metric('จำนวนข้อมูล', f'{len(df):,} แถว')
m2.metric('อัตราสอบผ่าน', f"{df['Passed'].mean():.1%}")
m3.metric('จำนวนฟีเจอร์', f'{len(NUM)+len(CAT)} ตัว')
m4.metric('โมเดลที่ดีที่สุด', best_name)

tab1, tab2, tab3 = st.tabs(['🔮 ทำนายผล', '📊 ภาพรวมข้อมูล', '📈 ประสิทธิภาพโมเดล'])

with tab1:
    st.subheader('กรอกข้อมูลการเรียน')
    c = st.columns(4)
    in_study = c[0].slider('ชั่วโมงอ่านหนังสือ/วัน', 0.0, 8.0, 3.0, 0.5)
    in_sleep = c[1].slider('ชั่วโมงนอน/วัน', 4.0, 10.0, 7.0, 0.5)
    in_abs   = c[2].slider('จำนวนครั้งการขาดเรียน', 0, 15, 3)
    in_att   = c[3].slider('เปอร์เซ็นต์เข้าเรียน', 40.0, 100.0, 85.0, 1.0)
    c2 = st.columns(4)
    in_quiz  = c2[0].slider('คะแนนควิซเฉลี่ย', 0, 100, 60)
    in_asn   = c2[1].slider('จำนวนงานที่ส่ง (จาก 10)', 0, 10, 7)
    in_gpa   = c2[2].slider('GPA เทอมก่อน', 0.0, 4.0, 2.5, 0.1)
    in_job   = c2[3].selectbox('ทำงานพาร์ทไทม์', ['No', 'Yes'])

    if st.button('🔮 ทำนายผล', type='primary'):
        X_in = pd.DataFrame([[in_study, in_sleep, in_abs, in_att, in_quiz, in_asn, in_gpa, in_job]],
                            columns=NUM + CAT)
        prob = float(model.predict_proba(X_in)[0][1])
        st.progress(prob)
        a, b = st.columns([1, 3])
        a.metric('โอกาสสอบผ่าน', f'{prob*100:.1f}%')
        if prob >= 0.7:   b.success('✅ แนวโน้มดี รักษาวิถีนี้ไว้!')
        elif prob >= 0.4: b.warning('⚠️ เริ่มเสี่ยง ลองเพิ่มชั่วโมงทบทวนบทเรียนอีกนิด')
        else:             b.error('🚨 เสี่ยงสูง! ควรปรึกษาอาจารย์และปรับแผนการเรียนด่วน')

with tab2:
    st.dataframe(df.head(100))
    i1, i2 = st.columns(2)
    with i1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(data=df, x='Passed', y='Study_Hours', palette='crest', ax=ax)
        ax.set_title('Study Hours vs Passed')
        st.pyplot(fig)
    with i2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=df, x='Part_Time_Job', y='Passed', palette='viridis', ax=ax)
        ax.set_ylim(0, 1); ax.set_title('Pass Rate by Part-Time Job')
        st.pyplot(fig)

with tab3:
    t1, t2 = st.columns(2)
    with t1:
        st.dataframe(res_df.round(4))
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(res_df)); w = 0.2
        for i, col in enumerate(res_df.columns):
            bars = ax.bar(x + i*w, res_df[col], w, label=col)
            ax.bar_label(bars, fmt='%.2f', fontsize=8)
        ax.set_xticks(x + 1.5*w); ax.set_xticklabels(res_df.index)
        ax.set_ylim(0.5, 1.1); ax.legend(ncol=2, loc='lower right')
        st.pyplot(fig)
    with t2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {best_name}')
        st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=fi, x='importance', y='feature', palette='crest', ax=ax)
    ax.set_title('Feature Importance (Random Forest)')
    st.pyplot(fig)