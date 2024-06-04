# -*- coding: utf-8 -*-
import os
import getpass

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error,r2_score
from tkinter import Tk, filedialog # for the file open and save dialog graphical user interface (GUI)
from pathlib import Path

from models import OneHotANN

user = getpass.getuser()

file_path = os.path.dirname(__file__)
os.chdir(file_path)

#%%

order_dict = {
    "< -2min": 1,
    "-2 to -1 min": 2,
    "-1 to -0.5 min": 3,
    "\u00B1 0.5 min": 4,
    "0.5 to 1 min": 5,
    "1 to 2 min": 6,
    "> 2 min": 7,
    }

def error_windows(value):
    if value < -2:
       window = "< -2min"
    elif value >= -2 and value < -1:
        window = "-2 to -1 min"
    elif value >= -1 and value <-0.5:
        window = "-1 to -0.5 min"
    elif value >= -0.5 and value <0.5:
        window = "\u00B1 0.5 min"
    elif value >= 0.5 and value <1:
        window = "0.5 to 1 min"
    elif value >= 1 and value <= 2:
        window = "1 to 2 min"
    else:
        window = "> 2 min"
       
    return window
    
def plot_bar(data, column, path, title='Bar Plot', order_dict=None, hue=None, save= False):
    counts = data[column].value_counts().reset_index()
    counts.columns = ['RT error window', 'count']
    if order_dict is not None:
        counts['Order'] = counts[column].replace(order_dict)
        counts.sort_values(by='Order', inplace=True)
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(counts['RT error window'], counts['count'])

    ax.set_title(title, fontsize=12)
    ax.bar_label(ax.containers[0], labels = [f"{x.get_height()} ({round(x.get_height()/len(data)*100, 1)} %)" for x in ax.containers[0]],
                           label_type='edge')
   
    
    if save == True:
        plt.savefig((path/title).with_suffix('.png'), bbox_inches='tight')
    
    plt.show()

def plot_scatter(data, x, y, hue, title='Title', save = False, path = None, dpi=500):

    lst1 = ['Train', 'Optimization', 'Validation', 'Test']
    lst2 = ['R2', 'MAE']
    metrics = {cat1: {cat2: "" for cat2 in lst2} for cat1 in lst1}
    
    for category in data[hue].unique():
        df = data[data[hue] == category]
        metrics[category]['R2'] = round(r2_score(df[x], df[y]),3)
        metrics[category]['MAE'] = round(mean_absolute_error(df[x], df[y]),3)
        
    min_rt = min(min(data[x]), min(data[y]))
    max_rt = max(max(data[x]), max(data[y]))
    
    plt.scatter(data[data[hue] == 'Train'][x], data[data[hue] == 'Train'][y],  
                alpha=0.1, color='red', label="Train ("+str(len(data[data[hue] == 'Train']))+")")
    plt.scatter(data[data[hue] == 'Optimization'][x], data[data[hue] == 'Optimization'][y],  
                alpha=0.1, color='black',label="Optimization ("+str(len(data[data[hue] == 'Optimization']))+")")
    plt.scatter(data[data[hue] == 'Validation'][x], data[data[hue] == 'Validation'][y],  
                alpha=0.1, color='green', label="Validation ("+str(len(data[data[hue] == 'Validation']))+")")
    plt.scatter(data[data[hue] == 'Test'][x], data[data[hue] == 'Test'][y],  
                alpha=0.8, label="Test ("+str(len(data[data[hue] == 'Test']))+")")
    plt.plot(data[x],data[x],'r')
    plt.legend(loc="upper left", title="All ("+str(len(data))+")", fontsize=8)
    plt.text(max_rt, min_rt, 
                ('Train $R^2$ = '+str("{:.3f}".format(metrics['Train']['R2']))  +'\n' +
                 'Validation $R^2$ = '+str("{:.3f}".format(metrics['Validation']['R2'])) +'\n' +
                 'Test $R^2$ = '+str("{:.3f}".format(metrics['Test']['R2'])) +'\n' +  
                 'Train MAE = '+str("{:.3f}".format(metrics['Train']['MAE']))+'\n'+
                 'Validation MAE = '+str("{:.3f}".format(metrics['Validation']['MAE']))  +'\n'+
                 'Test MAE = '+str("{:.3f}".format(metrics['Test']['MAE']))),  
                fontsize=8, 
                horizontalalignment="right", 
                verticalalignment="bottom")
    plt.title(title)
    plt.xlabel("Experimental  $t_R$ (min)")
    plt.ylabel("Predicted  $t_R$ (min)")
    if save == True:
        plt.savefig((path/title).with_suffix('.png'), bbox_inches='tight', dpi=dpi)
    plt.show()
       
root = Tk()
root.withdraw()
root.attributes("-topmost", True)

def open_file(title, initialdir = os.getcwd()):
    return filedialog.askopenfilename(initialdir=initialdir,
                                  filetypes=[('Excel*', '*.xlsx')], parent=root, title = title)


#%%

root_dir = Path(filedialog.askdirectory(title='Select or create project directory'))

name = str(root_dir).split('\\')[-1]

modelling_file = open_file('Select modelling data', initialdir = root_dir.parent)

#root_dir = pathlib.Path(file).parent.absolute()

modelling_data = pd.read_excel(modelling_file)
prediction_file = open_file('Select prediction data', initialdir = root_dir.parent)
prediction_data = pd.read_excel(prediction_file)
# Make Plots folder
plots_dir = root_dir/"Plots"
os.makedirs(plots_dir, exist_ok = True)

results_dir = root_dir/"Modelling Results"
os.makedirs(results_dir, exist_ok = True)


#%%

##############################

descriptors = ['logD', 'logP', 'nO', 'nC']
labs = modelling_data['LAB'].unique()

model = OneHotANN(dataset = modelling_data, index = 'Compound', X=descriptors, y='RT', 
                  one_hot = ['LAB', 'DrugClass'], max_layers = 2, interval = 10,
                  min_nodes = 50, max_nodes=200, iterations=5)
        
###############################

subsets = model.subsets
architecture_list = model.architecture_list

model.fit_ann(x_train = subsets['Train']['X'], 
              y_train=subsets['Train']['y'], 
              x_optimization = subsets['Optimization']['X'], 
              y_optimization = subsets['Optimization']['y'], 
              x_validation = subsets['Validation']['X'], 
              y_validation = subsets['Validation']['y'],
              x_test =  subsets['Test']['X'], 
              y_test = subsets['Test']['y'],
              architecture_list = architecture_list)

summary = model.fit_summary.sort_values(by=['Avg Validation MAE'], ascending=True)
best_architecture = model.best_architecture
predictions = model.predictions
removed_training_comp = model.single_class_instances

# Get the best predictions of the model and apply error windows
best_predictions = model.best_model_results
best_predictions["RT error window"] = best_predictions['Error'].apply(error_windows)

keys = []
for comp in best_predictions.index:
    keys.append(modelling_data[modelling_data['Compound'] == comp]['InChIKey'].values[0])

best_predictions.insert(0,'InChIKey', keys)

# Get the test set results
test_results = best_predictions[best_predictions['Set'] == 'Test']


plot_bar(test_results, column = 'RT error window', order_dict=order_dict,
         title=f"{name} Bar Plot (n = {len(test_results)})", save = True, path=plots_dir)

# Make new predictions

preds = model.make_new_predictions(prediction_data)[0]

# Get scatter plot of predictions for the best test set results

plot_scatter(best_predictions, x='RT', y='Mean RT', hue='Set', 
             title=name, save=True, 
             path = plots_dir)

# Get error window bar plot for the best test set results

plot_bar(test_results, column = 'RT error window', order_dict=order_dict,
         title=f"{name} Bar Plot (n = {len(test_results)})", 
         save = True, path=plots_dir)

for lab in labs:
        
    # Get best results for each laboratory
    lab_all = best_predictions[best_predictions['LAB'] == lab].reset_index()
    lab_test = lab_all[lab_all['Set'] == 'Test']
    
    lab_all.to_excel(results_dir/f"{lab} Modelling Results.xlsx", index=False)
    
    
    # Get error window bar plot for each laboratory
    half_min_method = len(lab_test[(lab_test['Error'] >= -0.5) & (lab_test['Error'] <=0.5)])

    plot_bar(lab_test, column = 'RT error window', order_dict = order_dict,
             title=f"{name} Bar Plot ({lab}, n = {len(lab_test)})", 
            save = True, path=plots_dir)

    
    # Get error boxplot and overlay swarmplot for each lab.
    
    bp = sns.boxplot(x='Error', y='Set', data=lab_all, fliersize = 5)
    bp.set_title(f"{name} ({lab})")
    bp.set_xlabel("Error (min)")
    bp.set_ylabel("")
    #bp.set_xticklabels(bp.get_xticklabels(), rotation = 90, ha="center", fontsize=8)
    plt.savefig(plots_dir/f"{name} - Error Boxplot ({lab}).png",dpi=500,
                bbox_inches='tight')


    bp = sns.swarmplot(x='Error', y='Set', data=lab_all, size=5)
    bp.set_title(f"{name} ({lab})")
    bp.set_xlabel("Error (min)")
    bp.set_ylabel("")
    #bp.set_xticklabels(bp.get_xticklabels(), rotation = 90, ha="center", fontsize=8)
    plt.savefig(plots_dir/f"{name} - Error Swarmplot ({lab}).png",dpi=500,
                bbox_inches='tight')


    # Get scatter plot for each lab
    plot_scatter(lab_all, x='RT', y='Mean RT', hue='Set', 
                 title=f'{name} ({lab})', save=True, 
                 path = plots_dir)

    
# Save all the files

preds.to_excel(root_dir/f"{name} - New Predictions.xlsx", index=False)
summary.to_excel(results_dir/f"{name} - Architecture Summary.xlsx", index=False)

 
param_list = ['X', 'train_split', 'optimisation_split', 'validation_split', 
              'test_split', 'stratify', 'max_layers', 'min_nodes', 'max_nodes', 
              'interval', 'iterations', 'patience', 'max_epochs', 'early_stop', 
              'batch_size', 'optimizer', 'activation', 'loss', 'random_state']

params =  {key:value for key, value in model.__dict__.items() if key in param_list}

with open(root_dir/f"{name} - Parameters.txt", 'w+') as f:
   
    f.write(f"Modelling Data: {modelling_file}\n"
            f"Prediction Data: {prediction_file}\n")
    f.write(f"Best architecture: {best_architecture}\n")
    for key, value in params.items():
        f.write(f"{key}: {value}\n")
    f.close()


