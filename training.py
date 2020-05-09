import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def create_ids(data,field,dict_field,i):
    # print(train.loc[i,'genres'])
    gen = eval(data.loc[i, field])
    field_val=[]
    if len(gen) != 0:
        for j in range(len(gen)):
            field_val.append(gen[0][dict_field])
    if field_val == []:
        field_val.append(0)
    return field_val

def create_dict_col(data,column):
    col=data[column].values
    d={}
    i=1
    new_col=[]
    #print(col)
    for val in col:
        if val not in d:
            d[val]=i
            i+=1
        new_col.append(d[val])
    #print(new_col)
    return d,new_col

def create_directors(data,i):
    # print(train.loc[i,'genres'])
    gen = eval(data.loc[i, 'crew'])
    director=[]
    if len(gen) != 0:
        for j in range(len(gen)):
            if gen[j]['job']=='Director':
                director.append(gen[j]['id'])

    return director

def preprocessing(filename):
    train = pd.read_csv(filename, sep='\t')
    pd.set_option('display.max_columns', None)
    #print(train.isna().sum())
    cols_to_drop = ['backdrop_path', 'homepage', 'imdb_id', 'poster_path', 'tagline', 'status', 'overview']
    train = train.drop(cols_to_drop, axis=1)
    genres = [[], [], []]
    gen_num = []
    languages = [[], [], []]
    lang_num = []
    companies = []
    prod_countries = []
    keywords = []
    director = [[], []]
    cast = [[], [], [], [], []]
    months = []
    years = []
    count = 0
    #print(train.describe())

    # print(train.head())

    for i in range(len(train)):
        date = train.loc[i, 'release_date'].split('-')
        if len(date) == 0:
            months.append(0)
            years.append(0)
        else:
            months.append(int(date[1]))
            years.append(int(date[0]))
        # print(train.loc[i,'belongs_to_collection'])
        if type(train.loc[i, 'belongs_to_collection']) == str:
            train.loc[i, 'belongs_to_collection'] = eval(train.loc[i, 'belongs_to_collection'])['id']
        else:
            train.loc[i, 'belongs_to_collection'] = 0
        if type(train.loc[i, 'genres']) == str:
            # print(train.loc[i,'genres'])
            gen_array = create_ids(train, 'genres', 'id',i)
            for j in range(3):
                if len(gen_array) <= j:
                    genres[j].append(0)
                else:
                    genres[j].append(gen_array[j])
            gen_num.append(len(gen_array))
        else:
            genres[0].append(0)
            genres[1].append(0)
            genres[2].append(0)
            gen_num.append(0)
        if type(train.loc[i, 'production_companies']) == str:
            companies.append(create_ids(train, 'production_companies', 'id',i)[0])
        else:
            companies.append(0)
        if type(train.loc[i, 'production_countries']) == str:
            prod_countries.append(create_ids(train, 'production_countries', 'iso_3166_1',i)[0])
        else:
            prod_countries.append(0)
        if type(train.loc[i, 'spoken_languages']) == str:
            lan_array = create_ids(train, 'spoken_languages', 'iso_639_1',i)
            for j in range(3):
                if len(lan_array) <= j:
                    languages[j].append(0)
                else:
                    languages[j].append(lan_array[j])
            lang_num.append(len(lan_array))

        else:
            for j in range(3):
                languages[j].append(0)
            lang_num.append(0)

        if type(train.loc[i, 'crew']) == str:
            dirs = create_directors(train,i)
            if dirs==[]:
                count+=1
            for j in range(2):
                if len(dirs) <= j:
                    director[j].append(0)
                else:
                    director[j].append(dirs[j])
        else:
            for j in range(2):
                director[j].append(0)
        if type(train.loc[i, 'cast']) == str:
            dirs = create_ids(train, 'cast', 'id',i)
            for j in range(5):
                if len(dirs) <= j:
                    cast[j].append(0)
                else:
                    cast[j].append(dirs[j])
        else:
            for j in range(5):
                cast[j].append(0)

    train['num_gen'] = gen_num
    #print('genres',len(train[train['genres']==0]))
    train['genres'] = genres[0]
    train['genres1'] = genres[1]
    train['genres2'] = genres[2]
    train['production_companies'] = companies
    #print('production_companies', len(train[train['production_companies'] == 0]))
    train['production_countries'] = prod_countries
    #print('production_countries', len(train[train['production_countries'] == 0]))
    train['spoken_languages'] = languages[0]
    #print('spoken_languages', len(train[train['spoken_languages'] == 0]))
    train['spoken_languages1'] = languages[1]
    train['spoken_languages2'] = languages[2]
    train['lang_num'] = lang_num
    train['director'] = director[0]
    train['director1'] = director[1]
    train['month'] = months
    train['year'] = years
    train['actor'] = cast[0]
    #print('actor', len(train[train['actor'] == 0]))
    train['actor1'] = cast[1]
    train['actor2'] = cast[2]
    train['actor3'] = cast[3]
    train['actor4'] = cast[4]
    cols_to_transform = ['original_language', 'production_countries', 'spoken_languages', 'spoken_languages1',
                         'spoken_languages2']
    #print('crew',count)
    for col in cols_to_transform:
        d, new_col = create_dict_col(train, col)
        train[col] = new_col

    cols_to_drop = ['crew', 'cast', 'title', 'original_title', 'Keywords', 'release_date', 'cast','vote_average']
    train = train.drop(cols_to_drop, axis=1)
    x = train.drop('revenue', axis=1)
    x['runtime'] = x['runtime'].fillna(0)

    #print(x)
    y = train['revenue']
    return x,y
def rmsle(estimator,x_test,y_true):
    """
    Calculates Root Mean Squared Logarithmic Error between two input vectors
    :param y_true: 1-d array, ground truth vector
    :param y_pred: 1-d array, prediction vector
    :return: float, RMSLE score between two input vectors
    """
    y_pred=estimator.predict(x_test)
    for i,val in enumerate(y_pred):
        if val<0:
            y_pred[i]=0
    assert y_true.shape == y_pred.shape, \
     ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
    return -1*np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))
if __name__ == '__main__':

    x,y=preprocessing('train.tsv')
    #print(x.columns)
    testx,testy=preprocessing('test.tsv')
    #print(train[train['revenue']<=100])
    grid={
        'n_estimators':[25,50,75,100,125],
        'learning_rate':[0.01,0.05,0.1,0.2],
        'max_depth':[3,4,6,8,10,12,15]
    }
    model = GradientBoostingRegressor()
    s=GridSearchCV(estimator=model,param_grid=grid,scoring=rmsle)
    s.fit(x,y)
    best_GBT=s.best_estimator_
    print(-1*s.best_score_)
    print(s.best_params_)
    params={'learning_rate':0.05,'max_depth':25,'n_estimators':100}
    model=GradientBoostingRegressor(learning_rate=params['learning_rate'],criterion='mae',max_depth=params['max_depth'],n_estimators=params['n_estimators'])
    model.fit(x,y)
    print(-1*rmsle(model,testx,testy))
    filename='GBTregressor.pkl'
    pickle.dump(best_GBT,open(filename,'wb'))
    print('GBT',model.score(testx,testy))
    grid={
        'n_estimators':[25,50,75,100,125,150],
        'max_depth':[8,10,12,14,15,20,25]
    }
    model = RandomForestRegressor()
    s = GridSearchCV(estimator=model, param_grid=grid, scoring=rmsle)
    s.fit(x,y)
    best_rf=s.best_estimator_
    print(-1*s.best_score_)
    print(s.best_params_)
    filename = 'RFregressor.pkl'
    pickle.dump(best_rf, open(filename, 'wb'))
