# import pickle
import gradio as gr
import numpy as np
import xgboost as xgb

model1=xgb.XGBRegressor()
model2=xgb.XGBRegressor()

model1.load_model('model1.json')
model2.load_model('model2.json')




def greed(Measure,Line_Item_Quantity,Line_Item_Value,Weight,Freight_Cost,Line_Item_Insurance):
    input_array=np.array([[Measure,Line_Item_Quantity,Line_Item_Value,Weight,Freight_Cost,Line_Item_Insurance]])
    pred1=model1.predict(input_array)
    pred2=model2.predict(input_array)
    pred1=float(np.asarray(pred1))
    pred2=float(np.asarray(pred2))
    return pred1,pred2



output1=gr.outputs.Textbox(label='Pack price')
output2=gr.outputs.Textbox(label='Unit price')

demo = gr.Interface(
    fn=greed,
    inputs=[gr.inputs.Number(),gr.inputs.Number(),gr.inputs.Number(),gr.inputs.Number(),gr.inputs.Number(),gr.inputs.Number()],
    outputs=[output1,output2],description='SHIPMENT PRICING PREDICTION')


demo.launch(share=True)