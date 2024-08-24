## 🌟 OBJECTIVE 🌟

The main task is to predict whether a patient has a **10-year risk** of developing coronary heart disease (CHD) 🩺🫀. This means determining the likelihood that a patient will develop CHD within the next 10 years based on specific factors. 

🔍 **Your Goal**: Build a predictive model using data on patients' demographics, behaviors, and medical history to classify them into one of two categories:

### 📊 Classification Categories:
- **🚨 High Risk (1)**: The patient is predicted to have a high likelihood of developing CHD within the next 10 years.
- **✅ Low Risk (0)**: The patient is predicted to have a low likelihood of developing CHD within the next 10 years.

By accurately predicting these categories, you can help in early intervention and potentially life-saving preventive measures! 💪❤️

## Variables Description

The dataset consists of the following attributes, categorized into demographic, behavioral, and medical factors:

### Demographic Factors

- **Sex**: The gender of the patient, coded as "M" (male) or "F" (female).
- **Age**: The age of the patient, recorded as a continuous variable.

### Behavioral Factors

- **is_smoking**: Whether the patient is a current smoker, recorded as "YES" or "NO."
- **Cigs Per Day**: The average number of cigarettes the patient smokes per day, treated as a continuous variable.

### Medical History Factors

- **BP Meds**: Whether the patient is on blood pressure medication (Nominal).
- **Prevalent Stroke**: Whether the patient has had a stroke in the past (Nominal).
- **Prevalent Hyp**: Whether the patient is hypertensive (Nominal).
- **Diabetes**: Whether the patient has diabetes (Nominal).

### Current Medical Factors

- **Tot Chol**: Total cholesterol level, recorded as a continuous variable.
- **Sys BP**: Systolic blood pressure, recorded as a continuous variable.
- **Dia BP**: Diastolic blood pressure, recorded as a continuous variable.
- **BMI**: Body Mass Index, recorded as a continuous variable.
- **Heart Rate**: The heart rate of the patient, treated as a continuous variable.
- **Glucose**: Glucose level, recorded as a continuous variable.

### Predictive Variable (Target)

- **10-year risk of CHD**: The target variable indicating whether the patient has a 10-year risk of coronary heart disease, coded as binary (1 for "Yes," 0 for "No").
