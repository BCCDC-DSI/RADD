import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.training.tracking.data_structures import NoDependency
import itertools
import time
#from tqdm import tqdm

class OneHotANN(Sequential):
    
    def __init__(self, dataset, index, X, y, one_hot, 
                 train_split=0.55, optimisation_split=0.15, validation_split=0.15, test_split=0.15, stratify=True,
                 max_layers=2, min_nodes=10, max_nodes=100, interval=10, iterations=5, patience=50, 
                 max_epochs=500, early_stop=True, batch_size=32, optimizer='adam', activation='relu', loss = 'mae', random_state = 0):

        self.dataset = dataset.dropna()
        self.columns_ = self.dataset.columns
        self.index = index
        self.X = X
        self.y = y
        self.one_hot = one_hot
        self.train_split = train_split
        self.optimisation_split = optimisation_split
        self.validation_split = validation_split
        self.test_split = test_split
        self.stratify = stratify
        self.max_layers = max_layers
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.interval = interval
        self.iterations = iterations
        self.patience = patience
        self.max_epochs = max_epochs
        self.early_stop = early_stop
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.activation = activation
        self.loss = loss
        self.random_state = random_state

        # Inherit from Keras classes (Sequential)
        
        super(OneHotANN, self).__init__()
        
        # Generate architecture list based on the mininum and maximum number of nodes, 
        # number of hidden layers and interval
        
        layers = []
        for layer in range(self.max_layers):
            if len(layers) == 0:
                layers.append(list(np.arange(self.min_nodes, self.max_nodes+self.interval,self.interval)))
            else:
                layers.append([None]+list(np.arange(self.min_nodes, self.max_nodes+self.interval,self.interval)))
        
        self.architecture_list=NoDependency([])
        invalid_architectures =[]
        all_architectures = list(itertools.product(*layers))
        
        for tup in all_architectures:
            for k in tup:
                 if (k == None and tup.index(k) == (len(tup)-1)):
                    self.architecture_list.append(list(filter(None.__ne__, list(tup))))
                    break
                 elif (k == None and tup[tup.index(k)+1] != None):
                    invalid_architectures.append(list(filter(None.__ne__, list(tup))))
                    break    
                 elif(not all(tup) == False):
                    self.architecture_list.append(list(filter(None.__ne__, list(tup))))
                    break
                 elif(all(k == None for k in tup[(tup.index(k)):(len(tup))]) == True):
                    self.architecture_list.append(list(filter(None.__ne__, list(tup))))
                    break

        self.subsets = NoDependency({'Train': {},
                   'Optimization': {},
                   'Validation': {},
                   'Test': {}
                   })
        
        used_cols = [self.index] + self.X + [self.y] + self.one_hot
        new_dataset = self.dataset[used_cols].set_index(self.index)

        if len(self.one_hot) == 2:
            new_dataset['Partitioner'] = new_dataset[self.one_hot[0]].astype(str) + "_" + new_dataset[self.one_hot[1]].astype(str)
        
        else:
            new_dataset['Partitioner'] =  new_dataset[self.one_hot[0]].astype(str)
        
        # Removes entries where there are single entries for the partitioner column  
        
        partitioner_counts = new_dataset['Partitioner'].value_counts()
        self.single_class_instances = new_dataset[new_dataset['Partitioner'].isin(partitioner_counts[partitioner_counts == 1].index)]
        new_dataset = new_dataset[~new_dataset['Partitioner'].isin(partitioner_counts[partitioner_counts == 1].index)]
                                  
# =============================================================================
#         self.new_dataset.reset_index(inplace=True)
#         new_dataset = new_dataset.set_index('Partitioner')
#         new_dataset = new_dataset.drop(self.single_class_instances['Partitioner'].values)
#         new_dataset.reset_index(inplace=True)
#         
#         new_dataset = new_dataset.set_index(self.index)
# =============================================================================
        
        self.classes_ = NoDependency({})
        
        for col in self.one_hot:
            self.classes_[col] = sorted(list(new_dataset[col].unique()))
            
        new_dataset = pd.get_dummies(new_dataset, columns=self.one_hot,drop_first=True)
        
        X = new_dataset.drop(self.y, axis=1)
        y = new_dataset[self.y]
        
        # Define splits
        # Works out how many entries are required to satisfy each split proportion
        
        entries = len(new_dataset)
        test_entries = entries*self.test_split
        optimisation_entries = entries*self.optimisation_split
        validation_entries = entries*self.validation_split
        test_entries = entries*self.test_split
        
        # Works out the test_size values needed for the subsequent splits
        
        second_split = validation_entries/(entries-test_entries)
        third_split = optimisation_entries/(entries - test_entries - optimisation_entries)
         
        try:
            X_int, X_test, y_int, y_test = train_test_split(self.X, self.y, test_size=self.test_split, random_state=self.random_state, stratify=X['Partitioner'])
        except:
            X_int, X_test, y_int, y_test = train_test_split(X, y, test_size=self.test_split, random_state=self.random_state)
        try:
            X_int2, X_val, y_int2, y_val = train_test_split(X_int, y_int, test_size=second_split, random_state=self.random_state, stratify=X_int["Partitioner"])
        except:   
            X_int2, X_val, y_int2, y_val = train_test_split(X_int, y_int, test_size=second_split, random_state=self.random_state)
        try:
            X_train, X_opt, y_train, y_opt = train_test_split(X_int2, y_int2, test_size=third_split, random_state = self.random_state, stratify=X_int2["Partitioner"])
        except:
            X_train, X_opt, y_train, y_opt = train_test_split(X_int2, y_int2, test_size=third_split, random_state = self.random_state)
       
        self.subsets['Train']['X'] = X_train
        self.subsets['Optimization']['X'] = X_opt
        self.subsets['Validation']['X'] = X_val
        self.subsets['Test']['X'] = X_test
        self.subsets['Train']['y'] = y_train
        self.subsets['Optimization']['y'] = y_opt
        self.subsets['Validation']['y'] = y_val
        self.subsets['Test']['y'] = y_test
        
    def fit_ann(self, x_train, y_train, x_optimization, y_optimization, x_validation, y_validation, x_test, y_test, architecture_list):
        self.name_ = y_train.name
        
        set_mapping = {'Train' : x_train,
            'Optimization': x_optimization,
            'Validation': x_validation,
            'Test': x_test
            }
        
        # Since we drop the "Partitioner" column we lose the category information for each entry.
        # Here we store those categories in the 'partition_labels' dictionary.
        
        partition_labels = {'Train': {},
                  'Optimization': {},
                  'Validation': {}, 
                  'Test' : {}
                  }
        
        for key, value in set_mapping.items():
            one_hots = value['Partitioner'].str.split("_")
            for cat in range(len(self.one_hot)):
                lst = []
                for entry in one_hots: 
                    lst.append(entry[cat])
                partition_labels[key][self.one_hot[cat]] = lst
                
        
        # Scale the different datasets

        self.scaler = StandardScaler()
        x_train_sc = self.scaler.fit_transform(x_train.drop('Partitioner', axis=1))
        x_optimization_sc = self.scaler.transform(x_optimization.drop('Partitioner', axis=1))
        x_validation_sc = self.scaler.transform(x_validation.drop('Partitioner', axis=1))
        x_test_sc = self.scaler.transform(x_test.drop('Partitioner', axis=1))
        
        
        
        self.input_nodes  = x_train_sc.shape[1]
        #max_layers = max([len(x) for x in architecture_list])
         
        avg_fit_time = []
        avg_epoch_stop = []
        avg_train_mae = []
        avg_train_r2 = []
        avg_validation_mae = []
        avg_validation_r2 = []
        avg_test_mae = []
        avg_test_r2 = []
        calc_params = []
    
        self.predictions = NoDependency({})
        
        for architecture in architecture_list:
            
            sub_dict = {}
            
            train_predictions_df = pd.DataFrame(y_train, index=x_train.index)
            optimization_predictions_df = pd.DataFrame(y_optimization, index=x_optimization.index)
            validation_predictions_df = pd.DataFrame(y_validation, index = x_validation.index)
            test_predictions_df = pd.DataFrame(y_test, index = x_test.index)
            
            for cat in self.one_hot:
                train_predictions_df[cat] = partition_labels['Train'][cat]
                optimization_predictions_df[cat] = partition_labels['Optimization'][cat]
                validation_predictions_df[cat] = partition_labels['Validation'][cat]
                test_predictions_df[cat] = partition_labels['Test'][cat]
            
            iteration_fit_time = []
            iteration_epoch_stop = []
            iteration_train_mae = []
            iteration_train_r2 = []
            iteration_validation_mae = []
            iteration_validation_r2 = []
            iteration_test_mae = []
            iteration_test_r2 = []
    
            for iteration in range(self.iterations):
                start_time = time.time()
                model_ = Sequential()
                for node in architecture:
                        model_.add(Dense(node, activation=self.activation))
                model_.add(Dense(1))
            
                model_.compile(optimizer=self.optimizer, loss=self.loss)
            
                if self.early_stop == True:
                    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=self.patience)
                else:
                    early_stop = None
                    
                model_.fit(x=x_train_sc, y=y_train.values, 
                    validation_data=(x_optimization_sc,y_optimization.values), batch_size=self.batch_size, epochs=self.max_epochs, callbacks=[early_stop], verbose=0)
                   
                iteration_fit_time.append(time.time()-start_time)
                iteration_epoch_stop.append(early_stop.stopped_epoch)  

                train_predictions = model_.predict(x_train_sc)
                optimization_predictions = model_.predict(x_optimization_sc)
                validation_predictions = model_.predict(x_validation_sc)
                test_predictions = model_.predict(x_test_sc)
                
                train_predictions_df[iteration+1] = train_predictions.reshape(-1)
                optimization_predictions_df[iteration+1] = optimization_predictions.reshape(-1)
                validation_predictions_df[iteration+1] = validation_predictions.reshape(-1)
                test_predictions_df[iteration+1] = test_predictions.reshape(-1)
                
                iteration_train_mae.append(round(mean_absolute_error(y_train, train_predictions),3))
                iteration_train_r2.append(round(r2_score(y_train, train_predictions),3))
        
                iteration_validation_mae.append(round(mean_absolute_error(y_validation, validation_predictions),3))
                iteration_validation_r2.append(round(r2_score(y_validation, validation_predictions),3))
                
                iteration_test_mae.append(round(mean_absolute_error(y_test, test_predictions),3))
                iteration_test_r2.append(round(r2_score(y_test, test_predictions),3))
                
                print("\rModel "+ str(architecture_list.index(architecture)+1) + " of "+ str(len(architecture_list))+ " ("+str(iteration+1)+" of "+ str(len(range(self.iterations)))+") completed")
        
            
            iteration_cols = [i+1 for i in range(self.iterations)]
            
            train_predictions_df['Mean '+self.name_] = train_predictions_df[iteration_cols].mean(axis=1)
            optimization_predictions_df['Mean '+self.name_] = optimization_predictions_df[iteration_cols].mean(axis=1)
            validation_predictions_df['Mean '+self.name_] = validation_predictions_df[iteration_cols].mean(axis=1)
            test_predictions_df['Mean '+self.name_] = test_predictions_df[iteration_cols].mean(axis=1)
        
            train_predictions_df['SD'] = train_predictions_df[iteration_cols].std(axis=1)
            optimization_predictions_df['SD'] = optimization_predictions_df[iteration_cols].std(axis=1)
            validation_predictions_df['SD'] = validation_predictions_df[iteration_cols].std(axis=1)
            test_predictions_df['SD'] = test_predictions_df[iteration_cols].std(axis=1)
            
            train_predictions_df['Error'] = train_predictions_df['Mean '+self.name_] - train_predictions_df[self.name_] 
            optimization_predictions_df['Error'] = optimization_predictions_df['Mean '+self.name_] - optimization_predictions_df[self.name_]
            validation_predictions_df['Error'] = validation_predictions_df['Mean '+self.name_] - validation_predictions_df[self.name_]
            test_predictions_df['Error'] = test_predictions_df['Mean '+self.name_] - test_predictions_df[self.name_]
            
            sub_dict['Train'] = train_predictions_df
            sub_dict['Optimization'] = optimization_predictions_df
            sub_dict['Validation'] = validation_predictions_df
            sub_dict['Test'] = test_predictions_df
            
            self.predictions[str(architecture)] = sub_dict
            
            avg_fit_time.append(np.asarray(iteration_fit_time).mean())
            avg_epoch_stop.append(np.asarray(iteration_epoch_stop).mean())
            avg_train_mae.append(np.asarray(iteration_train_mae).mean().round(3))
            avg_train_r2.append(np.asarray(iteration_train_r2).mean().round(3))
            avg_validation_mae.append(np.asarray(iteration_validation_mae).mean().round(3))
            avg_validation_r2.append(np.asarray(iteration_validation_r2).mean().round(3))
            avg_test_mae.append(np.asarray(iteration_test_mae).mean().round(3))
            avg_test_r2.append(np.asarray(iteration_test_r2).mean().round(3))
            calc_params.append(model_.count_params())
            
        columns = ["Layer "+str(x+1) for x in range(self.max_layers)]
        self.fit_summary = pd.DataFrame(architecture_list, columns = columns)
        self.fit_summary[['Avg Fit Time', 'Avg Stopped Epoch', 
                          'Avg Train MAE', 'Avg Train R2', 
                          'Avg Validation MAE', 'Avg Validation R2', 
                          'Avg Test MAE', 'Avg Test R2','Params']] = pd.DataFrame(list(zip(avg_fit_time, avg_epoch_stop, 
                                                                                   avg_train_mae, avg_train_r2, 
                                                                                   avg_validation_mae, avg_validation_r2,
                                                                                   avg_test_mae, avg_test_r2, calc_params))) 
       
        self.best_architecture = NoDependency(self.fit_summary[self.fit_summary['Avg Validation MAE'] == self.fit_summary['Avg Validation MAE'].min()][columns].dropna(axis=1).values.astype(int).tolist()[0])
         
        best_architecture_dict = dict(zip(['Layer '+str(x+1) for x in range(self.max_layers)], self.best_architecture))
        
        self.best_metrics = self.fit_summary.loc[np.all(self.fit_summary[list(best_architecture_dict)] == pd.Series(best_architecture_dict), axis=1)].to_dict('records')
        
        self.best_model_results = pd.DataFrame()
        for set_ in self.predictions[str(self.best_architecture)].keys():
            df = self.predictions[str(self.best_architecture)][set_]
            df['Set'] = set_
            self.best_model_results = pd.concat([self.best_model_results, df])
        
        # Refit model with best architecture
        
        print('Refitting model with best architecture: '+str(self.best_architecture))
        
        self.model_ = Sequential()
        for node in self.best_architecture:
            self.model_.add(Dense(node, activation=self.activation))
        self.model_.add(Dense(1))
            
        self.model_.compile(optimizer=self.optimizer, loss=self.loss)
    
        self.model_.fit(x=x_train_sc, y=y_train.values, 
            validation_data=(x_optimization_sc,y_optimization.values), batch_size=self.batch_size, epochs=self.max_epochs, callbacks=[early_stop], verbose=0)
                     
    
    def plot_architecture_results(self, architecture):
       
        self.architecture_dict = dict(zip(['Layer '+str(x+1) for x in range(self.max_layers)], architecture))
        
        self.results = self.predictions[str(architecture)]
        
        self.metrics_ = self.fit_summary.loc[np.all(self.fit_summary[list(self.architecture_dict)] == pd.Series(self.architecture_dict), axis=1)].to_dict('records')[0]
       
        arch = ":".join([str(self.input_nodes)]+[str(item) for item in architecture]+[str(1)])
        min_val = []
        max_val = []
        for key in self.results.keys():
            min_val.append(self.results[key][[self.name_, 'Mean '+self.name_]].min().min())
            max_val.append(self.results[key][[self.name_, 'Mean '+self.name_]].max().max())
        
        min_val = min(min_val)
        max_val = max(max_val)
                 
        plt.scatter(self.results['Train'][self.name_],self.results['Train']['Mean '+self.name_], color='red', label = 'Train', alpha=0.2)
        plt.scatter(self.results['Validation'][self.name_],self.results['Validation']['Mean '+self.name_], color='green', label = 'Validation',alpha=0.2)
        plt.scatter(self.results['Test'][self.name_],self.results['Test']['Mean '+self.name_], label = 'Test', alpha=0.5)
        plt.plot(self.results['Test'][self.name_], self.results['Test'][self.name_])
        plt.text(max_val+0.4, min_val, 
                ('Train R2 = '+ str(self.metrics_['Avg Train R2']) +'\n' +
                 'Validation R2 = '+str(self.metrics_['Avg Validation R2']) +'\n' +
                 'Test R2 = '+str(self.metrics_['Avg Test R2']) +'\n' +  
                 'Train MAE = '+str(self.metrics_['Avg Train MAE']) +'\n'+
                 'Validation MAE = '+ str(self.metrics_['Avg Validation MAE']) +'\n'+
                 'Test MAE = '+ str(self.metrics_['Avg Test MAE'])),  
                fontsize=8, 
                horizontalalignment="right", 
                verticalalignment="bottom")
        
        plt.title(arch)
        plt.xlabel("Actual "+self.name_)
        plt.ylabel("Predicted "+self.name_)
        plt.legend(loc="upper left", fontsize=8)
        plt.xlim(min_val-1, max_val+1)
        plt.ylim(min_val-1, max_val+1)
    
        plt.show()
    
    def make_new_predictions(self, dataset):
# =============================================================================
#         self.prediction_dataset = dataset.copy()
#         self.return_cols =  self.prediction_dataset.drop(self.X, axis=1).reset_index(drop=True)
#         self.prediction_dataset = self.prediction_dataset[self.X+self.one_hot[1:]]
#         self.pred_control = ""
#         
#         
#         for one_hot_class in self.one_hot:
#             self.removed_data = pd.DataFrame()
#           
#             if one_hot_class not in self.prediction_dataset.columns:
#                 self.pred_control += one_hot_class
#                 self.pred_control_cols = [f'{self.pred_control}_{x}' for x in self.classes_[self.pred_control]]
#                 for x in self.pred_control_cols[1:]:
#                     self.prediction_dataset[x] = 0
#             elif one_hot_class in dataset.columns:
#                 self.prediction_dataset = self.prediction_dataset[self.prediction_dataset[one_hot_class].isin(self.classes_[one_hot_class])].reset_index(drop=True)
#                 self.removed_data = pd.concat([self.removed_data, self.prediction_dataset[~self.prediction_dataset[one_hot_class].isin(self.classes_[one_hot_class])].reset_index(drop=True)])
#                 self.prediction_dataset = pd.get_dummies(self.prediction_dataset, columns=[one_hot_class], drop_first=True)
#                 
#         
#         preds = []
#         pred_dfs = []
#         
#         for x in self.pred_control_cols:
#             pred_df = self.prediction_dataset.copy()
#             if self.pred_control_cols.index(x) != 0:
#                 pred_df[x] = 1
#                 pred_dfs.append(pred_df.copy())
#                 pred_df = self.scaler.transform(pred_df)
#                 preds.append(self.model_.predict(pred_df))
#             else:
#                 pred_dfs.append(pred_df.copy())
#                 pred_df = self.scaler.transform(pred_df)
#                 preds.append(self.model_.predict(pred_df))
#         preds = pd.DataFrame(np.concatenate(preds, axis=1))
#         preds = preds.mask(preds < 0, 0)
#         preds.columns = self.pred_control_cols
#         
#         preds = pd.concat([self.return_cols, preds], axis=1)
#          
#         return preds, pred_df
# =============================================================================

        one_hot_classes = ['LAB_'+s for s in self.classes_[self.one_hot[0]]]
        self.prediction_dataset = dataset
        self.removed_data = self.prediction_dataset[~self.prediction_dataset[self.one_hot[1]].isin(self.classes_[self.one_hot[1]])]
        self.prediction_dataset = self.prediction_dataset[self.prediction_dataset[self.one_hot[1]].isin(self.classes_[self.one_hot[1]])].reset_index(drop=True)
        
        self.return_cols = self.prediction_dataset.drop(self.X, axis=1).reset_index(drop=True)
        
        self.prediction_dataset = self.prediction_dataset[self.X+self.one_hot[1:]]

        for x in one_hot_classes[1:]:
            self.prediction_dataset[x] = 0
        self.prediction_dataset = pd.get_dummies(self.prediction_dataset, columns=self.one_hot[1:], drop_first=True)
        preds = []
        pred_dfs = []
        for x in one_hot_classes:
            pred_df = self.prediction_dataset.copy()
            if one_hot_classes.index(x) != 0:
                pred_df[x] = 1
                pred_dfs.append(pred_df.copy())
                pred_df = self.scaler.transform(pred_df)
                preds.append(self.model_.predict(pred_df))
            else:
                pred_dfs.append(pred_df.copy())
                pred_df = self.scaler.transform(pred_df)
                preds.append(self.model_.predict(pred_df))
        preds = pd.DataFrame(np.concatenate(preds, axis=1))
        preds = preds.mask(preds < 0, 0)
        preds.columns = one_hot_classes
        
        preds = pd.concat([self.return_cols, preds], axis=1)
        
        return preds, pred_dfs
    
    def save_model(self, fname):
        self.model_.save(fname,
                         save_format = 'h5')
        #del self.model_
        
    def load_model(self, fname):
        self.model_ = load_model(fname)
        
