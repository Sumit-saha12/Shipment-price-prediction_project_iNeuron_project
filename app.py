
import gradio as gr
import pickle
import numpy as np


# with open("/home/sumit/coding/shipment_prediction_project/model1.pkl", "rb") as f:
#         clf1  = pickle.load(f)

# model=pickle.load(open('model2.pkl','rb'))

with open("/home/sumit/coding/shipment_prediction_project/model1.pkl", "rb") as f:
        clf2  = pickle.load(f)

a1= gr.Number(label='Unit of Measure (Per Pack)')
a2= gr.Number(label='Line Item Quantity')
a3= gr.Number(label='Line Item Value')
a4= gr.Number(label='Weight (Kilograms)')
a5= gr.Number(label='Freight Cost (USD)')
a6= gr.Number(label='Line Item Insurance (USD)')

def predict_pack_price(a1,a2,a3,a4,a5,a6):
    input_array=np.array([[a1,a2,a3,a4,a5,a6]])
    pred=clf2.predict(input_array)
    return pred

# def predict_unit_price(a1,a2,a3,a4,a5,a6):
#     input_array=np.array([[a1,a2,a3,a4,a5,a6]])
#     pred=clf2.predict(input_array)
#     return pred

outputs1=gr.outputs.Textbox()
outputs2=gr.outputs.Textbox()
demo1 =gr.Interface(fn=predict_pack_price,inputs=[a1,a2,a3,a4,a5,a6],outputs=outputs1,description='PACK PRICE')
# demo2 =gr.Interface(fn=predict_unit_price,inputs=[a1,a2,a3,a4,a5,a6],outputs=outputs2,description="UNIT PRICE")

demo1.launch(share=True)
# demo2.launch()



