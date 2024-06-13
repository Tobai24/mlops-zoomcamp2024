import pickle


with open('/app/model.bin', 'rb') as f_model:
    model = pickle.load(f_model)

with open('/app/model2.bin', 'rb') as f_vectorizer:
    vectorizer = pickle.load(f_vectorizer)


def predict_duration(feature):
    input_data = vectorizer.transform(feature)
    prediction = model.predict(input_data)
    return prediction

mean_predicted_duration = predict_duration()
print(f"Mean predicted duration for May 2023: {mean_predicted_duration}")
