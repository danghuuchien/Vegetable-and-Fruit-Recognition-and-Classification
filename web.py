import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
#Tensorflow Model Prediction
def model_prediction (test_image):
    #mô hình adam
    model = tf.keras.models.load_model("trained_model_90.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr) 
    return np.argmax(predictions) #return index of max element
# Định nghĩa hàm đọc nhãn từ tệp labels.txt

def read_labels():
    with open("C:\HocMay\App_Web\labels.txt") as f:
        content = f.readlines()
    labels = [i.strip() for i in content]
    return labels

# Định nghĩa hàm vẽ biểu đồ cột
def plot_prediction_results(results, labels):
    prediction_counts = {}
    for _, prediction in results:
        if prediction in prediction_counts:
            prediction_counts[prediction] += 1
        else:
            prediction_counts[prediction] = 1
    
    # Lọc ra những nhãn có kết quả dự đoán và sắp xếp theo số lần dự đoán
    sorted_predictions = sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True)
    top_predictions = sorted_predictions[:10]
    
    # Lấy tên và số lần dự đoán cho 10 cột đầu tiên
    predicted_labels = [label for label, _ in top_predictions]
    predicted_counts = [count for _, count in top_predictions]
    
    # Tạo biểu đồ cột cho 10 cột đầu tiên
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(predicted_labels, predicted_counts)
    ax.set_xlabel('Loại')
    ax.set_ylabel('Số lần dự đoán')
    ax.set_ylim(0, 10)  
    ax.set_title('Biểu đồ dự đoán')
    ax.set_xticklabels(predicted_labels, rotation=0)
    
    # Ẩn các nhãn trên đỉnh của các cột
    for p in ax.patches:
        ax.annotate('', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
    ax.set_yticks(range(1, 11))
    
    st.pyplot(fig)


#Sider bar
st.sidebar.title("Dashboard")
app_mode= st.sidebar.selectbox("Chọn nội dung", ["Home","About Project","Prediction"])
#Main Page
if(app_mode=="Home"):
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "FRUITS & VEGETABLES.png"
    st.image(image_path, width=500)
#About Project
elif(app_mode=="About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("Bộ dữ liệu gồm 36 loại rau củ quả: ")
    st.code("fruits: banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables: cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, caulifl")
    st.subheader("Content")
    st.text("Dữ liệu ba gồm 3 thư mục:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

elif app_mode == "Prediction":
    st.header("Dự đoán")
    uploaded_files = st.file_uploader("Tải ảnh lên:", accept_multiple_files=True)
    labels = read_labels()
    
    if uploaded_files:
        st.write(f"Tổng số ảnh được tải lên: {len(uploaded_files)}")
        results = []
        for uploaded_file in uploaded_files[:10]:
            image = Image.open(uploaded_file)
            image.thumbnail((50, 50))
            st.image(image, caption=uploaded_file.name, width=200, use_column_width=200)
            result_index = model_prediction(uploaded_file)
            results.append((uploaded_file.name, labels[result_index]))
        
        if st.button("Predict"):
            st.write("Kết quả dự đoán:")
            for file_name, prediction in results:
                st.success(f"Ảnh {file_name}: Dự đoán là {prediction}")
            # Vẽ biểu đồ dự đoán
            st.header("Biểu đồ")
            plot_prediction_results(results, labels)