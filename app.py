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

st.set_page_config(page_title='Student Pass Predictor', page_icon='🎓',
                   layout='wide', initial_sidebar_state='expanded')
sns.set_theme(style='whitegrid')

# ---------- ตกแต่งเมนูซ้ายให้เหมือน ML HUB ----------
st.markdown("""
<style>
[data-testid="stSidebar"] .stRadio > div {gap: .4rem;}
[data-testid="stSidebar"] .stRadio label {
    border: 1px solid transparent; border-radius: .8rem;
    padding: .55rem 1rem; cursor: pointer; width: 100%;
}
[data-testid="stSidebar"] .stRadio label:hover {background: rgba(99,102,241,.15);}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(90deg, rgba(99,102,241,.35), rgba(99,102,241,.08));
    border-color: #6366f1;
}
[data-testid="stSidebar"] input[type="radio"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('##### ML HUB NAVIGATION')
    st.divider()
    menu = st.radio('เลือกเมนู', ['🏠 หน้าหลัก', '👨‍ ผู้พัฒนา'], label_visibility='collapsed')
    st.divider()
    st.caption('ระบบทำนายโอกาสสอบผ่านของนักศึกษา')

    with st.container(border=True):
        st.markdown('**📌 เกี่ยวกับโปรเจค**')
        st.markdown('เปรียบเทียบโมเดล Machine Learning 3 ชนิด เพื่อทำนายโอกาสสอบผ่านจากพฤติกรรม'
                    'การเรียน 8 ด้าน แล้วเลือกโมเดลที่แม่นยำที่สุดมาให้บริการ')
        st.markdown('**🎯 วิธีใช้งาน**')
        st.markdown('1. เปิดแท็บ **🔮 ทำนายผล**  \n2. เลื่อนแถบกรอกข้อมูลของคุณ  \n3. กดปุ่มทำนายเพื่อดูผลและคำแนะนำ')

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
    return df, pipes, best, res_df, cm, fi

df, pipes, best_name, res_df, cm, fi = build_all()
base_rate = df['Passed'].mean()

# ================= หน้าหลัก =================
if menu == '🏠 หน้าหลัก':
    st.title('🎓 ระบบทำนายโอกาสสอบผ่านของนักศึกษา')
    st.caption('มินิโปรเจค | Student Performance Prediction | ข้อมูล 1,500 รายการ')

    th_feat = {'Study_Hours':'ชั่วโมงอ่านหนังสือ','Sleep_Hours':'ชั่วโมงนอน','Absences':'การขาดเรียน',
               'Attendance_Pct':'เปอร์เซ็นต์เข้าเรียน','Quiz_Score':'คะแนนควิซ','Assignments':'การส่งงาน',
               'GPA_Prev':'GPA เทอมก่อน','Part_Time_Job_Yes':'ทำงานพาร์ทไทม์','Part_Time_Job_No':'ไม่ทำงานพาร์ทไทม์'}
    s1, s2, s3 = st.columns(3)
    s1.markdown(f'**🤖 โมเดลที่ใช้งาน:** {best_name}')
    s2.markdown(f"**🎯 ความแม่นยำ:** {res_df.loc[best_name, 'Accuracy']:.1%}")
    s3.markdown(f"**🔑 ปัจจัยที่มีผลที่สุด:** {th_feat.get(fi.iloc[0]['feature'], fi.iloc[0]['feature'])}")
    st.divider()

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
        in_model = st.selectbox('🤖 เลือกโมเดลที่ใช้ทำนาย', list(pipes.keys()),
                                index=list(pipes.keys()).index(best_name),
                                help='ค่าเริ่มต้นคือโมเดลที่แม่นยำที่สุด — ลองสลับโมเดลแล้วเปรียบเทียบผลดูได้')

        if st.button('🔮 ทำนายผล', type='primary'):
            st.session_state['pred_input'] = {
                'study': in_study, 'sleep': in_sleep, 'abs': in_abs, 'att': in_att,
                'quiz': in_quiz, 'asn': in_asn, 'gpa': in_gpa, 'job': in_job,
                'model': in_model}

        if 'pred_input' in st.session_state:
            d = st.session_state['pred_input']

            st.divider()
            st.subheader('📋 ทวนข้อมูลที่คุณกรอก')
            r1, r2, r3, r4 = st.columns(4)
            r1.metric('อ่านหนังสือ', f"{d['study']:.1f} ชม./วัน")
            r2.metric('นอนหลับ', f"{d['sleep']:.1f} ชม./วัน")
            r3.metric('ขาดเรียน', f"{d['abs']} ครั้ง")
            r4.metric('เข้าเรียน', f"{d['att']:.0f}%")
            r5, r6, r7, r8 = st.columns(4)
            r5.metric('คะแนนควิซ', f"{d['quiz']} คะแนน")
            r6.metric('ส่งงาน', f"{d['asn']}/10 ชิ้น")
            r7.metric('GPA เทอมก่อน', f"{d['gpa']:.2f}")
            r8.metric('พาร์ทไทม์', 'ทำ' if d['job'] == 'Yes' else 'ไม่ทำ')

            X_in = pd.DataFrame([[d['study'], d['sleep'], d['abs'], d['att'],
                                  d['quiz'], d['asn'], d['gpa'], d['job']]],
                                columns=NUM + CAT)
            prob = float(pipes[d['model']].predict_proba(X_in)[0][1])

            st.divider()
            st.subheader(f"🎯 ผลการทำนาย (โมเดล: {d['model']})")
            st.progress(prob)
            a, b = st.columns([1, 3])
            a.metric('โอกาสสอบผ่าน', f'{prob*100:.1f}%', delta=f'{(prob-base_rate)*100:+.1f}%')
            if prob >= 0.7:   b.success('✅ แนวโน้มดี รักษาวิถีนี้ไว้!')
            elif prob >= 0.4: b.warning('⚠️ เริ่มเสี่ยง ลองเพิ่มชั่วโมงทบทวนบทเรียนอีกนิด')
            else:             b.error('🚨 เสี่ยงสูง! ควรปรึกษาอาจารย์และปรับแผนการเรียนด่วน')

            advice = []
            if d['study'] < 2:  advice.append('เพิ่มชั่วโมงอ่านหนังสือเป็นอย่างน้อย 2 ชม./วัน')
            if d['sleep'] < 6:  advice.append('นอนให้พอ 6–8 ชม. เพื่อความจำและสมาธิที่ดีขึ้น')
            if d['abs'] > 5:    advice.append('ลดการขาดเรียน ไม่ควรเกิน 2–3 ครั้งต่อเทอม')
            if d['att'] < 70:   advice.append('เข้าเรียนให้สม่ำเสมอมากขึ้น (เป้าหมาย > 80%)')
            if d['quiz'] < 50:  advice.append('ทบทวนบทเรียนเพิ่มเติมก่อนสอบควิซครั้งถัดไป')
            if d['asn'] < 6:    advice.append('ส่งงานที่ค้างให้ครบทุกชิ้น')
            if d['job'] == 'Yes' and prob < 0.5: advice.append('จัดสมดุลเวลาทำงานพาร์ทไทม์กับเวลาเรียนใหม่')
            if not advice:      advice.append('พฤติกรรมการเรียนดีทุกด้าน รักษามาตรฐานนี้ไว้!')
            st.markdown('**💡 คำแนะนำเฉพาะบุคคล:**')
            for adv in advice:
                st.markdown(f'- {adv}')

    with tab2:
        st.download_button('⬇️ ดาวน์โหลดชุดข้อมูล (CSV)', df.to_csv(index=False).encode('utf-8-sig'),
                           file_name='student_performance.csv', mime='text/csv')
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

# ================= หน้าผู้พัฒนา =================
else:
    st.title('👨‍ ผู้พัฒนา')
    st.caption('ทีมงานมินิโปรเจควิชา Machine Learning')

    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        with c1:
            photo_url = 'https://drive.google.com/thumbnail?id=1VHoUraXA_sfFoSOcl53_ZoGnhUEB_qJf&sz=w1000'
            st.markdown(f'''
            <div style="text-align:center; padding-top: 10px;">
              <img src="{photo_url}"
                   style="width:140px; height:140px; border-radius:50%; object-fit:cover;
                          border:3px solid #6366f1; box-shadow:0 0 18px rgba(99,102,241,.45)">
            </div>''', unsafe_allow_html=True)
        with c2:
            st.markdown('### นายทินภัทร ช้อยสามนาค')
            st.markdown('**รหัสนักศึกษา:** 664245011  \n'
                        '**สาขาวิชา:** วิทยาการคอมพิวเตอร์  \n'
                        '**คณะ/ชั้นปี:** คณะวิทยาศาสตร์และเทคโนโลยี / ปีที่ 4')

    with st.container(border=True):
        st.markdown('### 📮 ช่องทางติดต่อ')
        st.markdown('**อีเมล:** 664245011@webmail.npru.ac.th  \n'
                    '**GitHub:** [664245011-cmd (คลิกเพื่อดู Repositories)](https://github.com/664245011-cmd?tab=repositories)')

    with st.container(border=True):
        st.markdown('### 🛠️ เทคโนโลยีที่ใช้')
        st.write('Python • Pandas • Scikit-learn • Matplotlib • Streamlit')

st.divider()
st.caption('จัดทำโดย นายทินภัทร ช้อยสามนาค (664245011) | มินิโปรเจควิชา Machine Learning')