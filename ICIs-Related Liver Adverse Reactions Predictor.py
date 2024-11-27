import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('XGBoost.pkl')

# 定义特征选项
area_options = {
    0: 'Africa (0)',
    1: 'Americas (1)',
    2: 'Asia (2)',
    3: 'Europe (3)',
    4: 'Oceania (4)',
    5: 'Other (5)'
}
cancer_types_options = {
    0: 'Adrenocortical cancer (0)', 1: 'Bladder cancer (1)', 2: 'Brain cancer (2)', 3: 'Breast cancer (3)',
    4: 'Cervix cancer (4)', 5: 'Cholangiocarcinoma (5)', 6: 'Colorectal cancer (6)', 7: 'Endometrial cancer (7)',
    8: 'Esophageal carcinoma (8)', 9: 'Gastric cancer (9)', 10: 'Glioma (10)', 11: 'Head and neck cancer (11)',
    12: 'Leukaemia (12)', 13: 'Lung cancer (13)', 14: 'Lymphoma (14)', 15: 'Melanoma (15)', 16: 'Mesothelioma (16)',
    17: 'Other (17)', 18: 'Ovarian cancer (18)', 19: 'Pancreatic cancer (19)', 20: 'Prostate cancer (20)',
    21: 'Renal cancer (21)', 22: 'Sarcoma (22)', 23: 'Skin cancer (23)', 24: 'Thymoma (24)', 25: 'Thyroid carcinoma (25)',
    26: 'Urothelial cancer (26)'
}
drug_options = {
    0: 'Atezolizumab (0)', 1: 'Avelumab (1)', 2: 'Cemiplimab (2)', 3: 'Durvalumab (3)', 4: 'Nivolumab (4)', 5: 'Pembrolizumab (5)'
}
yes_no_options = {0: 'No (0)', 1: 'Yes (1)'}

# 定义特征名
feature_names = [
    "Age", "Sex", "Area", "Cancer types", "Drug", "Hypertension", "Diabetes", "Autoimmune disease", 
    "NSAIDs", "PPI", "Statin", "CTLA-4", "Chemo", "TTD", "Anti-hypertension drugs"
]

# Streamlit用户界面
st.title("ICIs-Related Liver Adverse Reactions Predictor")  # 设置应用标题

# 输入字段：用户数据
age = st.selectbox("Age:", options=[0, 1], format_func=lambda x: '18-64 (0)' if x == 0 else '≥65 (1)')
sex = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')
area = st.selectbox("Area:", options=list(area_options.keys()), format_func=lambda x: area_options[x])
cancer_type = st.selectbox("Cancer types:", options=list(cancer_types_options.keys()), format_func=lambda x: cancer_types_options[x])
drug = st.selectbox("Drug:", options=list(drug_options.keys()), format_func=lambda x: drug_options[x])
hypertension = st.selectbox("Hypertension:", options=list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x])
diabetes = st.selectbox("Diabetes:", options=list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x])
autoimmune_disease = st.selectbox("Autoimmune disease:", options=list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x])
nsaids = st.selectbox("NSAIDs:", options=list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x])
ppi = st.selectbox("PPI:", options=list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x])
statin = st.selectbox("Statin:", options=list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x])
ctla4 = st.selectbox("CTLA-4:", options=list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x])
chemo = st.selectbox("Chemo:", options=list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x])
ttd = st.selectbox("TTD:", options=list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x])
anti_hypertension_drugs = st.selectbox("Anti-hypertension drugs:", options=list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x])

# 将输入转换为特征值
feature_values = [age, sex, area, cancer_type, drug, hypertension, diabetes, autoimmune_disease, nsaids, ppi, statin, ctla4, chemo, ttd, anti_hypertension_drugs]
features = np.array([feature_values])

# 当用户点击预测按钮时，进行预测
if st.button("Predict"):
    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model，you have a high risk of ICIs-related liver adverse reactions. "
            f"The model predicts that your probability of developing ICIs-related liver adverse reactions is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a relevant specialist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of ICIs-related liver adverse reactions. "
            f"The model predicts that your probability of not developing ICIs-related liver adverse reactions is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your liver health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )
    st.write(advice)

    # 计算SHAP值并显示力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
