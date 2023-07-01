import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO


CLASS_NAMES = ['Cataracts', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']


def main():
    st.title('EYE DISEASE CLASSIFICATION')
    model = load_model()
    image = load_image()
    result = st.button('PREDICT EYE DISEASE')
    if result:
        st.write('Calculating results...')

        predicted_class, accuracy_percentage = predict(model, CLASS_NAMES, image)

        # if accuracy_percentage >= 0.80:
        st.sidebar.write('Predicted Class:', predicted_class)
        st.sidebar.write('Probability: ', accuracy_percentage,'%')
        # else:
        #     st.sidebar.write('**Predicted Class: **', predicted_class)
        #     st.sidebar.write('**Probability: **', accuracy_percentage,'%')




def load_model():
    model = tf.keras.models.load_model('model/VGG19/model_test.tflite')
    return model


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)

        temp = Image.open(BytesIO(image_data))
        print('dwd', temp)

        return Image.open(BytesIO(image_data))
    else:
        return None


def predict(model, class_names, image):
     # Resizing the image
    resized_image = image.resize((224, 224))

    # Convert PIL image into array
    image = np.array(resized_image)

    # # Normalize the image
    normalized_image = image / 255.0

    input_image = np.expand_dims(normalized_image, axis=0)

    predictions = model.predict(input_image)
    print(predictions)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    accuracy = np.max(predictions[0])
    accuracy_percentage = '{:.2}'.format(accuracy)


    return predicted_class, float(accuracy_percentage)








if __name__ == '__main__':
    main()

# st.title("EYE DISEASE CLASSIFICATION")
# st.write('\n')

# MODEL = tf.keras.models.load_model("./model/VGG19/model.epoch06-loss0.34.h5")

# image = Image.open('images/10007_right.jpeg')
# show = st.image(image, use_column_width=True)

# st.sidebar.title("Upload Image")


# #Disabling warning
# st.set_option('deprecation.showfileUploaderEncoding', False)
# #Choose your own image
# uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )

# if uploaded_file is not None:
#     u_img = Image.open(uploaded_file)
#     show.image(u_img, 'Uploaded Image', use_column_width=True)
    
#     image = np.asarray(u_img)/255
#     my_image= np.resize(image, (224,224))



# # For newline
# st.sidebar.write('\n')
    
# if st.sidebar.button("Click Here to Classify"):
    
#     if uploaded_file is None:
        
#         st.sidebar.write("Please upload an Image to Classify")
    
#     else:
        
#         with st.spinner('Classifying ...'):
            
#             prediction = predict(cat_clf["w"], cat_clf["b"], my_image)
#             time.sleep(2)
#             st.success('Done!')
            
#         st.sidebar.header("Algorithm Predicts: ")
        
#         probability = "{:.3f}".format(float(prediction*100))
        
        
#         if prediction > 0.5:
#             st.sidebar.write("It's a 'Cat' picture.", '\n' )
#             st.sidebar.write('**Probability: **',probability,'%')
#             st.sidebar.audio(audio_bytes)
                             
#         else:
#             st.sidebar.write(" It's a 'Non-Cat' picture ",'\n')
#             st.sidebar.write('**Probability: **',probability,'%')
    
    
    


# @st.cache(allow_output_mutation=True)
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_png_as_page_bg(png_file):
#     bin_str = get_base64_of_bin_file(png_file) 
#     page_bg_img = '''
#     <style>
#     .stApp {
#     background-image: url("data:image/png;base64,%s");
#     background-size: cover;
#     background-repeat: no-repeat;
#     background-attachment: scroll; # doesn't work
#     }
#     </style>
#     ''' % bin_str
    
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#     return

# # set_png_as_page_bg('/content/background.webp')

# upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
# c1, c2= st.columns(2)
# if upload is not None:
#     im= Image.open(upload)
#     img= np.asarray(im)
#     image= cv2.resize(img,(224, 224))
#     img= preprocess_input(image)
#     img= np.expand_dims(img, 0)
#     c1.header('Input Image')
#     c1.image(im)
#     c1.write(img.shape)


# #load weights of the trained model.
# vgg_model.load_weights('')

# # prediction on model
# vgg_preds = vgg_model.predict(img)
# vgg_pred_classes = np.argmax(vgg_preds, axis=1)
# c2.header('Output')
# c2.subheader('Predicted class :')
# c2.write(classes[vgg_pred_classes[0]] )