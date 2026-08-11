import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Student Pass Predictor', page_icon='🎓')
model = pickle.load(open('best_model.pkl', 'rb'))

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

st.markdown('---')
st.subheader('📊 ประสิทธิภาพโมเดล')
st.image('results.png', use_column_width=True)
