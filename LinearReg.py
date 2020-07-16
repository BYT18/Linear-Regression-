import pandas as pd
from sklearn import linear_model
import tkinter as tk 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import statsmodels.api as sm

df = pd.read_csv('r2test.csv')

print (df.shape)
print(df.head())

columns = df.columns

feature_cols = columns[0:4]

#print(feature_cols)



X = df[feature_cols] # Features

Y = df[columns[8]]

print (X)
print (Y)

#plt.scatter(df[columns[3]], df[columns[8]], color='red')
#plt.title('Stock Index Price Vs Interest Rate', fontsize=14)
#plt.xlabel('Interest Rate', fontsize=14)
#plt.ylabel('Stock Index Price', fontsize=14)
#plt.grid(True)
#plt.show()

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)


# tkinter GUI
root= tk.Tk()

canvas1 = tk.Canvas(root, width = 500, height = 400)
canvas1.pack()

# with sklearn
Intercept_result = ('Intercept: ', regr.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
canvas1.create_window(260, 220, window=label_Intercept)

# with sklearn
Coefficients_result  = ('Coefficients: ', regr.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
canvas1.create_window(260, 240, window=label_Coefficients)

# New_Interest_Rate label and input box
label1 = tk.Label(root, text=' Oil_WTI Value: ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

# New_Unemployment_Rate label and input box
label2 = tk.Label(root, text=' NGP_NYMX Value: ')
canvas1.create_window(120, 120, window=label2)

entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)

#Feature 3
label3 = tk.Label(root, text=' NEQTY_S&P500 INDX_USD Value: ')
canvas1.create_window(120, 140, window=label3)

entry3 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 140, window=entry3)

#Feature 4
label4 = tk.Label(root, text=' Basis: ')
canvas1.create_window(120, 160, window=label4)

entry4 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 160, window=entry4)

def values(): 
    global f1 #our 1st input variable
    f1 = float(entry1.get()) 
    
    global f2 #our 2nd input variable
    f2 = float(entry2.get()) 
    
    global f3 #our 2nd input variable
    f3 = float(entry3.get()) 

    global f4 #our 2nd input variable
    f4 = float(entry4.get()) 

    Prediction_result  = ('Predicted FX Rate: ', regr.predict([[f1,f2,f3,f4]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)
    
button1 = tk.Button (root, text='Predict FX Rate',command=values, bg='orange') # button to call the 'values' command above 
canvas1.create_window(270, 200, window=button1)
 
#plot 1st scatter 
figure3 = plt.Figure(figsize=(5,4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(df[columns[2]].astype(float),df[columns[8]].astype(float), color = 'r')
scatter3 = FigureCanvasTkAgg(figure3, root) 
scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax3.legend(['Stock_Index_Price']) 
ax3.set_xlabel('Interest Rate')
ax3.set_title('Interest Rate Vs. Stock Index Price')

#plot 2nd scatter 
figure4 = plt.Figure(figsize=(5,4), dpi=100)
ax4 = figure4.add_subplot(111)
ax4.scatter(df[columns[1]].astype(float),df[columns[8]].astype(float), color = 'g')
scatter4 = FigureCanvasTkAgg(figure4, root) 
scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax4.legend(['Stock_Index_Price']) 
ax4.set_xlabel('Unemployment_Rate')
ax4.set_title('Unemployment_Rate Vs. Stock Index Price')

figure5 = plt.Figure(figsize=(5,4), dpi=100)
ax5 = figure5.add_subplot(111)
ax5.scatter(df[columns[0]].astype(float),df[columns[8]].astype(float), color = 'g')
scatter5 = FigureCanvasTkAgg(figure5, root) 
scatter5.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax5.legend(['Stock_Index_Price']) 
ax5.set_xlabel('Unemployment_Rate')
ax5.set_title('Unemployment_Rate Vs. Stock Index Price')

root.mainloop()