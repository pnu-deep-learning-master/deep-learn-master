#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.models import load_model

import tkinter

import plotly.offline as offline
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# 학습한 모델 불러오기

model = load_model('./RNA_model.h5')

# 실행 파일 GUI

window=tkinter.Tk()
window.title("Predict Cpf1")
window.geometry("420x240+100+100")
window.resizable(True, True)

def close():
    global window
    global t
    global RNA
    t = int(t_entry.get())
    RNA = R_entry.get()
    window.destroy()

t = 10
RNA = 'ACGT'

t_entry = tkinter.Entry(window,width =10)
t_entry.place(x=280,y=95)

t_label=tkinter.Label(window, text = '몇 등 까지의 데이터를 출력할까요?')
t_label.place(x=60,y=90)

R_entry = tkinter.Entry(window,width =10)
R_entry.place(x=280,y=35)

R_label=tkinter.Label(window, text = '테스트할 RNA를 입력하세요')
R_label.place(x=60,y=30)
R1_label=tkinter.Label(window, text = '(대문자로 입력해주세요)')
R1_label.place(x=60,y=50)

Q_button = tkinter.Button(window, text ='결과 출력', width=15, command=close)
Q_button.place(x=150, y = 160)

window.mainloop()

# RNA를 onehot-encoding로 전처리 하는 과정

def onehot_encoding_34bp(bp):
    
    ACGT = [['A'], ['C'], ['G'], ['T']]
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    onehot_encoder.fit(ACGT)

    RNA_encoded_bp = []

    bp = np.array(bp)

    for i in range(len(bp)):
        bp_raw = np.array(list(bp[i])).reshape(-1, 1)
        bp_encoded = onehot_encoder.transform(bp_raw)


        RNA_encoded_bp.append(bp_encoded)

    return RNA_encoded_bp


# 테스트할 RNA가 있는 txt 파일 불러오기 / txt 파일 최상위에는 bp_34가 있어야만 한다.
# onehot-encoding 과정을 거쳐 A = [1,0,0,0], C = [0,1,0,0], G = [0,0,1,0], T = [0,0,0,1] 의 형태로 변환한다

test_rna = []
test_rna.append(RNA)

test_rna_encoded_34bp = onehot_encoding_34bp(test_rna)
rna_test = test_rna_encoded_34bp
rna_test = np.array(rna_test)

# i = 추후 반복문에서 사용될 변수 / a = 입력받는 RNA(bp) 숫자 
# n = 입력받은 bp에서 34개 단위로 구분하여 생기는 갯수 / t = 최상위권 정렬 시 출력하는 갯수 ex. t = 10 이면 top10이 출력된다.

i = 0
a = len(rna_test[0])
n = a - 33


# 변환된 RNA를 34개 단위로 자르는 과정 / [1 ~ 34], [2 ~ 35] .... 같은 형식으로 자른다.

rna_test_tmp = rna_test[0]
rna_test_sort = []

for i in range(n):
    rna_test_sort.append(np.array(rna_test_tmp[i:i+34]))

rna_test_sort = np.array(rna_test_sort)

# predict 함수를 사용하여 변환한 RNA를 넣어 Indel 값을 예상한다.

rna_result = model.predict(rna_test_sort)
rna_result = np.array(rna_result)


# flatten을 사용하여 1차원으로 만들어준다

rna_result_f = rna_result.flatten()


# 그래프의 X축을 위하여 onehot-encoding 되지않은 RNA를 34개 단위로 자른다.

rna = test_rna
rna = np.array(rna)

rna_tmp = rna[0]
rna_34 = []
rna_34_1 = []


for i in range(n):
    rna_34.append(np.array(rna_tmp[i:i+34]))
    rna_34_1.append(np.array(rna_tmp[i]+" "+str(i+1)+"~"+str(i+34)))
  

rna_34 = np.array(rna_34)
rna_34_1 = np.array(rna_34_1)

# 결과 값들을 정렬하기 위하여 기존 분리되어있는 결과값과 변환되지 않은 RNA 값을 하나의 배열로 만든다.


result_array = []
for i in range(n):
    result_array.append([rna_result_f[i],rna_34[i]])

# txt 파일로 입력된 순서대로 나온 결과 값을 저장한다. 

np.savetxt('./test_result.txt',result_array,fmt='%s') 

result_array.sort()
result_array.reverse()
result_sort = np.array(result_array)
result_sort_f = result_sort.flatten()

# txt 파일로 정렬된 결과 값을 저장한다. 
np.savetxt('./test_result_sort.txt',result_sort,fmt='%s') 

# 한개의 배열이 된 결과값을 정렬한 배열에서 최상위권을 입력된 값만큼 자른다. ex. top 10의 경우 정렬된 배열에서 상위 10개를 뽑아낸다.

result_sort_x = []
result_sort_y = []

for i in range(n):
    result_sort_x.append(np.array(result_sort_f[2*i+1]))
    result_sort_y.append(np.array(result_sort_f[2*i]))

result_sort_x = np.array(result_sort_x)
result_sort_y = np.array(result_sort_y)

result_top_x = []
result_top_y = []
result_top = []


result_top_x.append(result_sort_x[:t])
result_top_y.append(result_sort_y[:t])

for i in range(t):
    result_top.append(result_sort[i])

result_top_x = np.array(result_top_x)
result_top_y = np.array(result_top_y)
result_top = np.array(result_top)

result_top_x = result_top_x.flatten()
result_top_y = result_top_y.flatten()
result_top_f = result_top.flatten()

result_top_r_x = []
result_top_r_y = []

for i in range(t):
    result_top_r_x.append(result_top_f[len(result_top_f)-(1+i)*2])
    result_top_r_y.append(result_top_f[len(result_top_f)-1-(i)*2])
    
result_top_r_x = np.array(result_top_r_x)
result_top_r_y = np.array(result_top_r_y)

# 결과를 각각의 그래프와 표를 사용하여 하나의 HTML 파일로 저장한다. 

x=rna_34_1
y=rna_result_f

x_s=result_sort_x
y_s=result_sort_y

y_t=result_top_r_x
x_t=result_top_r_y

fig = make_subplots(rows=2, cols=2,
    specs=[[{"type": "bar"},None],
           [{"type": "bar"},{"type": "table"}]]
)

fig.add_trace(go.Bar(x=x, y=y,text=rna_34, xaxis="x2", yaxis="y2",
                     name='기본 차트'),row=1,col=1)
fig.add_trace(go.Bar(x=y_t, y=x_t,text=y_t,orientation='h', xaxis="x2", yaxis="y2",
                     name='상위권 차트'),row=2,col=1)
fig.add_trace(go.Table(header=dict(values=['RNA Sequence', 'Indel Frequency'], align="left"),
                       cells=dict(values= [result_top_x,result_top_y], align = "left")),
                row=2, col=2
)
fig.layout.xaxis.update({'title' : 'RNA Sequence'})
fig.layout.xaxis.update({'domain' : [1,1]})
fig.layout.yaxis.update({'title' : 'Indel Frequency'})

fig.layout.xaxis2.update({'title' : 'Indel Frequency'})
fig.layout.yaxis2.update({'title' : 'RNASequence'})

fig.update_layout(showlegend=True,
    title_text=("Prediction of CRISPR-Cpf1 RNA Sequence: "+ RNA),
    font=dict(size=12)
)

offline.plot(fig, filename = './graph.html')
