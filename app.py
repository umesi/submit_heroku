from flask import Flask, render_template, request,Response,make_response
from werkzeug.datastructures import FileStorage
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
import re
import copy
import xgboost as xgb
import pickle
import memory


app = Flask(__name__)


sort_list=memory.sort_list
drop_list=memory.drop_list
count_dict=memory.count_dict

stock={}

with open('model.pickle', mode='rb') as fp:
    che= pickle.load(fp)


@app.route('/')
def index():
    return render_template('csv.html')

@app.route('/',methods=["POST"])
def hoge():
    csv_data = request.files['send_data']
    # csvファイルのみ受け付ける
    if isinstance(csv_data, FileStorage) and csv_data.content_type == 'text/csv':
        df_1 = pd.read_csv(csv_data).reindex(columns=sort_list)
        df=memory.func(df_1)
        df=xgb.DMatrix(df)
        pred=che.predict(df,ntree_limit=632)
        pred[pred<0]=0
        stock["df_1"]=df_1["お仕事No."]
        stock["pred"]=pred

        return render_template('download.html')
    else:
        raise ValueError('data is not csv')

@app.route("/export")
def export_file():
    submission=pd.DataFrame({"お仕事No.":stock["df_1"],
                "応募数 合計":stock["pred"]
                })
    resp = make_response(submission.to_csv(index=False))
    resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

if __name__ == "__main__":
    app.run(debug=False,host='127.0.0.1', port=9999)