import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
import pdb
import matplotlib.colors as mcolors
import matplotlib.patches as patches
# 加载数据集


# 准备特征数据和目标变量
data=pd.read_excel('/Users/yuanzhisun/Desktop/NICE/data/rawdataNoP.xlsx',sheet_name='Sheet4')
#print([   data['sio2']  , data['al2o3']  ,data['fe2o3']  , data['cao'] ,data['mgo'] , data['so3'] , data['tio2'] , data['k2o'] ,  data['na2o']  , data['p2o5']])

# 创建 MinMaxScaler 对象
scaler = MinMaxScaler(feature_range=(0, 1))



# 将数据进行归一化处理

df1=data.iloc[:,0]
ddf1=df1.values
normalized_1 = scaler.fit_transform(ddf1.reshape(-1,1))
min_value1 = np.min(data.iloc[:, 0])
max_value1 = np.max(data.iloc[:, 0])
nd1=pd.DataFrame(normalized_1)

df2=data.iloc[:,1]
ddf2=df2.values
normalized_2 = scaler.fit_transform(ddf2.reshape(-1,1))
min_value2 = np.min(data.iloc[:, 1])
max_value2 = np.max(data.iloc[:, 1])
nd2=pd.DataFrame(normalized_2)

df3=data.iloc[:,2]
ddf3=df3.values
normalized_3 = scaler.fit_transform(ddf3.reshape(-1,1))
min_value3 = np.min(data.iloc[:, 2])
max_value3 = np.max(data.iloc[:, 2])
nd3=pd.DataFrame(normalized_3)

df4=data.iloc[:,3]
ddf4=df4.values
normalized_4 = scaler.fit_transform(ddf4.reshape(-1,1))
min_value4 = np.min(data.iloc[:, 3])
max_value4 = np.max(data.iloc[:, 3])
nd4=pd.DataFrame(normalized_4)

df5=data.iloc[:,4]
ddf5=df5.values
normalized_5 = scaler.fit_transform(ddf5.reshape(-1,1))
min_value5 = np.min(data.iloc[:, 4])
max_value5 = np.max(data.iloc[:, 4])
nd5=pd.DataFrame(normalized_5)

df6=data.iloc[:,5]
ddf6=df6.values
normalized_6 = scaler.fit_transform(ddf6.reshape(-1,1))
min_value6 = np.min(data.iloc[:, 5])
max_value6 = np.max(data.iloc[:, 5])
nd6=pd.DataFrame(normalized_6)

df7=data.iloc[:,6]
ddf7=df7.values
normalized_7 = scaler.fit_transform(ddf7.reshape(-1,1))
min_value7 = np.min(data.iloc[:, 6])
max_value7 = np.max(data.iloc[:, 6])
nd7=pd.DataFrame(normalized_7)

df8=data.iloc[:,7]
ddf8=df8.values
normalized_8 = scaler.fit_transform(ddf8.reshape(-1,1))
min_value8 = np.min(data.iloc[:, 7])
max_value8 = np.max(data.iloc[:, 7])
nd8=pd.DataFrame(normalized_8)

df9=data.iloc[:,8]
ddf9=df9.values
normalized_9 = scaler.fit_transform(ddf9.reshape(-1,1))
min_value9 = np.min(data.iloc[:, 8])
max_value9 = np.max(data.iloc[:, 8])
nd9=pd.DataFrame(normalized_9)

df10=data.iloc[:,14]
ddf10=df10.values
normalized_10 = scaler.fit_transform(ddf10.reshape(-1,1))
min_value10 = np.min(data.iloc[:, 14])
max_value10 = np.max(data.iloc[:, 14])
nd10=pd.DataFrame(normalized_10)

df11=data.iloc[:,15]
ddf11=df11.values
normalized_11 = scaler.fit_transform(ddf11.reshape(-1,1))
min_value11 = np.min(data.iloc[:, 15])
max_value11 = np.max(data.iloc[:, 15])
nd11=pd.DataFrame(normalized_11)


dfft=data.iloc[:,12]
ddfft=dfft.values
normalized_ft = scaler.fit_transform(ddfft.reshape(-1,1))
min_valueft = np.min(data.iloc[:, 12])
max_valueft = np.max(data.iloc[:, 12])
ndft=pd.DataFrame(normalized_ft)


nd = np.column_stack((nd1,nd2,nd3,nd4,nd5,nd6,nd7,nd8,nd9,nd10,nd11,ndft))

nd_ultra=pd.DataFrame(nd)
#划分输入与输出
X = nd_ultra.iloc[:,0:11]
y = nd_ultra.iloc[:,11]


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 定义 XGBoost 回归模型
#xgb_model = xgb.XGBRegressor(learning_rate=0.0418606990609245,
 #                           n_estimators=288,
  #                         max_depth=10,
   #                          min_child_weight=0.036630667572042946,
    #                      colsample_bytree=0.955525771867695, subsample=0.6244766064569987,
     #                        random_state=42)


xgb_model = xgb.XGBRegressor(random_state=42)
# 训练模型
xgb_model.fit(X_train, y_train)

# 在训练集和测试集上进行预测
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

original_ytestpred = y_test_pred * (max_valueft - min_valueft) + min_valueft
original_ytest = y_test * (max_valueft - min_valueft) + min_valueft
original_ytrain = y_train * (max_valueft - min_valueft) + min_valueft
original_ytrainpred = y_train_pred * (max_valueft - min_valueft) + min_valueft

# 评估模型
mse_train = mean_squared_error(original_ytrain, original_ytrainpred)
mse_test = mean_squared_error(original_ytest, original_ytestpred)

print("Train MSE:", mse_train)
print("Test MSE:", mse_test)


feature_names=['SiO2','Al2O3','Fe2O3','CaO','MgO','SO3','TiO2','K2O','Na2O','SiO2/Al2O3','Acid/Base']
X.columns = feature_names
# 绘制特征重要性条形图
#plt.figure(figsize=(8, 6))  # 调整图形大小
#plt.barh(range(len(feature_importance)), feature_importance, color='royalblue')

# 去除图形的边框和网格线
#plt.box(False)
#plt.grid(False)

# 调整坐标轴标签字体大小和颜色
#plt.yticks(range(len(feature_importance)), feature_names, fontsize=12, color='black')
#plt.xticks(fontsize=12, color='black')

# 设置标题和坐标轴标签
#plt.title("Feature Importance", fontsize=14, fontweight='bold', color='black')
#plt.xlabel("Importance", fontsize=12, fontweight='bold', color='black')
#plt.ylabel("Feature", fontsize=12, fontweight='bold', color='black')
#plt.show()
#plt.savefig('/Users/yuanzhisun/Desktop/NICE/featureimportance1.png', dpi=600)

import shap

explainer=shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(X)

#sample_index=222
shap.initjs()
#shap.force_plot(explainer.expected_value,shap_values[1,:],X.iloc[1,:],matplotlib=True)




#shap.summary_plot(shap_values, X, plot_type="bar",show=False)
# Customize the appearance of the plot
fig, ax = plt.gcf(), plt.gca()
#fig.set_size_inches(10, 6)  # Adjust the figure size



# Define a custom colormap with a gradient of purple shades from bottom to top
num_bars = len(ax.patches)
colors = plt.cm.Purples_r(np.linspace(0.6, 0.2, num_bars))  # Change the values (0.6, 0.2) for different shades

# Set the colors for the bars as a gradient of purple from bottom to top
for i, bar in enumerate(ax.patches):
    bar.set_facecolor(colors[i])
    bar.set_edgecolor('black')
# Set the title and axis labels

plt.xlabel("SHAP Value")

# Show the plot
#plt.show()
plt.savefig('/Users/yuanzhisun/Desktop/NICE/NoP2O5/MeanSHAPALLcharacteristics.png', dpi=1000)



#plt.savefig('/Users/yuanzhisun/Desktop/NICE/NoP2O5/shapplot-WithExtra.png', dpi=1000)

#shap.summary_plot(shap_values, X,show=False)
plt.xticks(fontsize=12, color='black')
#plt.savefig('/Users/yuanzhisun/Desktop/NICE/NoP2O5/shapplot-WithExtra.png', dpi=1000)
#plt.show()






#十折交叉验证法计算每一折的RMSE


# 进行10折交叉验证并获取每次验证的指标
#rmse_scores = np.sqrt(-cross_val_score(xgb_model, X, y, cv=10, scoring='neg_mean_squared_error'))
#r2_scores = cross_val_score(xgb_model, X, y, cv=10, scoring='r2')



# 打印每次验证的RMSE和R2指标
#print("Cross-validation RMSE scores:", rmse_scores)
#print("Cross-validation R2 scores:", r2_scores)


kfold = KFold(n_splits=10, shuffle=True, random_state=42)
y_pred = cross_val_predict(xgb_model, X, y, cv=kfold)

original_ytest = y_pred * (max_valueft - min_valueft) + min_valueft
original_y = y * (max_valueft - min_valueft) + min_valueft


rmse = np.sqrt(mean_squared_error(original_y, original_ytest))
r2 = r2_score(original_y, original_ytest)
rmse_avg = np.mean(rmse)
r2_avg = np.mean(r2)



plt.scatter(original_y, original_ytest, c='purple', edgecolors='black', alpha=1, linewidths=1, marker='o')
plt.plot([min(original_y), max(original_y)], [min(original_y), max(original_y)], '--', color='red')

bbox_props = dict(boxstyle='round,pad=0.5', facecolor='none', edgecolor='purple')
plt.text(min(original_y), max(original_y), f"Average RMSE: {rmse_avg:.2f} \n   Average R2: {r2_avg:.2f}",bbox=bbox_props,ha='left', va='top')


plt.xlabel('Exact values')
plt.ylabel('Predicted values')
plt.title('Performance diagram for XGBoost')
plt.show()
#plt.savefig('/Users/yuanzhisun/Desktop/NICE/NoP2O5/performance-XGBoost.png', dpi=1000)