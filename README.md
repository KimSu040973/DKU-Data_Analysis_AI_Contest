# DKU-(SDSS_천체)Data_Analysis_AI_Contest
2020-2학기 천체 데이터 분석 AI 경진대회  
[단국대 소·중 데이터 분석 AI 경진대회](https://dacon.io/competitions/official/235638/overview/description)    



## 1. EDA & Feature engineering
-   사칙연산과 도메인 지식을 통한 다양한 변수 생성으로 초기 Feature_ver.1 생성  
-   생성한 모든 변수를 사용하여 lightgbm 기반 Permutation Importance 진행  
-   Permutation Importance를 기준으로 365개 변수에서 125개 변수로 축소 
-   아래와 같은 다양한 모델을 사용하여 스태킹을 했으나 성능의 한계를 느낌 
-   성능 향상을 위해 더 넓은 도메인 지식을 사용하여 Feature_ver.2 생성  
## 2. Modeling
-   모든 모델은 StratifiedKfold를 사용하여 정확도 검증
-   처음에는 LightGBM 단일 모델만을 사용하였으나 test accuracy가 0.9366에서 멈춤  
-   이후 스태킹을 위해 XGBoost, RandomForest, SVM, LogisticRegression, AdaBoost의 모델들을 시도  
-   GridSearchCV와 RandomizedSearch, HyperOpt 등의 Parameter 튜닝 모델들을 사용하여 개별 모델들의 성능을 올리고 스태킹을 시도하였으나 train 데이터에 대해 과적합이 확인되며 오히려 성능이 떨어짐  
-   Feature_ver.1과 Feature_ver.2 이외에도 ver.3, ver.4를 만들어 Feature별 스태킹 및 Model별 스태킹 또한 진행하였으나 심한 과적합을 보임   
-   이후 Feature_ver.1과 Feature_ver.2에만 모델을 사용하여 여러 조합을 통해 가장 높은 CV값을 보이는 모델 제출  
## 3. 한계점
- EDA를 사용하여 보다 잘 구분되는 파생변수를 생성하였으나 효과가 없는 변수가 더 많았음  
- 전체 데이터(320000)에서 LightGBM과 같은 좋은 모델들이 구분하지 못하는 약 6.1%의 데이터를 LogisticRegression이 약 13.5% 구분한다는 것을 확인하였으나 제대로 접목시키지 못함  
- 산점도 내에서 변수간의 관계 파악을 통해 변수 생성을 할 수 있었으나 시간상 초기 생성한 360개 변수에 대해서 모두 산점도를 그려보지 못함  

### feature ver.1
- raw data의 변수들의 조합을 이용하여 사칙연산 피쳐 생성  
- www.sdss.org 에서 domain 관련 변수 추가  
- LightGBM 기반 Permutation Importance를 적용하여 365개 변수에서 240개 제거  

```python
#이상치 제거
for i in range(len(fea)):
    trn=trn[trn[fea[i]]>np.min(tst[fea[i]], axis=0)]
    trn=trn[trn[fea[i]]<np.max(tst[fea[i]], axis=0)]
    
#설명변수와 반응변수 분리
trn_target = trn['class']
trn = trn.drop('class', axis=1)

for df in [trn,tst]:
    # 옵저브 디텍트 연속형으로 전환
    df['nObserve']=df['nObserve'].astype('float')
    df['nDetect']=df['nDetect'].astype('float')

    #카테고리별 max, min, max-min, std, sum을 구한다.
    #max-min
    df['max-min'] = df[all_fea].max(axis=1)-df[all_fea].min(axis=1)
    df['max-min_ugriz'] = df[ugriz].max(axis=1)-df[ugriz].min(axis=1)
    df['max-min_dered'] = df[dered].max(axis=1)-df[dered].min(axis=1)
    #std
    df['std'] = df[all_fea].std(axis=1)
    df['std_ugriz'] = df[ugriz].std(axis=1)
    df['std_dered'] = df[dered].std(axis=1)
    #파장별 합
    df['sum'] = df[all_fea].sum(axis=1)
    df['sum_ugriz'] = df[ugriz].sum(axis=1)
    df['sum_dered'] = df[dered].sum(axis=1)
    #파장별 최대값
    df['max'] = df[all_fea].max(axis=1)
    df['max_ugriz'] = df[ugriz].max(axis=1)
    df['max_dered'] = df[dered].max(axis=1)
    #파장별 최소값
    df['min'] = df[all_fea].min(axis=1)
    df['min_ugriz'] = df[ugriz].min(axis=1)
    df['min_dered'] = df[dered].min(axis=1)
    #파장별 max-max,min=min,sum-sum
    df['max-max']=df[ugriz].max(axis=1)-df[dered].max(axis=1)
    df['min-min']=df[ugriz].min(axis=1)-df[dered].min(axis=1)
    df['sum-sum']=df[ugriz].sum(axis=1)-df[dered].sum(axis=1)

    #왜도,첨도 구하기
    df['skew']=skew(df[ugriz],axis=1)
    df['kurtosis']=kurtosis(df[ugriz],axis=1)
    df['dered_skew']=skew(df[dered],axis=1)
    df['dered_kurtosis']=kurtosis(df[dered],axis=1)
    df['airmass_skew']=skew(df[airmass],axis=1)
    df['airmass_kurtosis']=kurtosis(df[airmass],axis=1)

    #조합으로 연산 피쳐 생성
    for c1,c2 in tqdm(itertools.combinations(fea2,2)):
        dif_col=f'diff_{c1}_{c2}'
        div_col=f'div_{c1}_{c2}'
        sum_col=f'sum_{c1}_{c2}'
        mul_col=f'mul_{c1}_{c2}'
        df[dif_col]=df[c1]-df[c2]
        df[div_col]=df[c1]/df[c2]
        df[sum_col]=df[c1]+df[c2]
        df[mul_col]=df[c1]*df[c2]

    #EDA를 통해 생성한 피쳐
    df['redshift%14'] = df['redshift']%14
    df['log_redshift']=np.log1p(df['redshift'])
    df['log_redshift']=df['log_redshift'].fillna(0)

    #도메인에서 얻은 파생변수 생성
    #출처: https://www.sdss.org/dr16/algorithms/segue_target_selection/#Legacy
    df['l-color'] = (-0.436*df['u']) + (1.129*df['g']) - (0.119*df['r']) - (0.574*df['i']) + (0.1984)
    df['s-color'] = (-0.249*df['u']) + (0.794*df['g']) - (0.555*df['r']) + (0.234)
    df['P1'] = (0.91*(df['u']-df['g'])) + (0.415*(df['g']-df['r'])) - (1.280)
     # 소수점 4자리 까지만 나타내는 asinh 변수 생성
    df['asinh_mu'] = -2.5/np.log(10)*(np.arcsinh(df.u/24.63/(2.8e-10))-22.689378693319245)
    df['asinh_mg'] = -2.5/np.log(10)*(np.arcsinh(df.g/25.11/(1.8e-10))-23.131211445598282)
    df['asinh_mr'] = -2.5/np.log(10)*(np.arcsinh(df.r/24.80/(2.4e-10))-22.843529373146502)
    df['asinh_mi'] = -2.5/np.log(10)*(np.arcsinh(df.i/24.36/(3.6e-10))-22.43806426503834)
    df['asinh_mz'] = -2.5/np.log(10)*(np.arcsinh(df.z/22.83/(1.48e-09))-21.024370929730330)

trn['class'] = trn_target
```



### feature ver.2 :
- 연산피쳐 줄이고 도메인 관련 변수를 추가적으로 생성
- redshift가 Random Forest 분류기로 Feature Importance에서 중요하다고 나와 redshift 관련 연산변수 추가


```python
ftr=trn.drop("class",axis=1)
target=trn['class']

rf_clf = RandomForestClassifier(n_estimators = 500, 
                                random_state=9697,
                                verbose=True,
                                oob_score=True,
                                n_jobs=-1)
rf_clf.fit(ftr, target)
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   20.3s
[Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:  1.6min
[Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:  3.7min
[Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:  4.2min finished
RandomForestClassifier(n_estimators=500, n_jobs=-1, oob_score=True,
                       random_state=9697, verbose=True)

mp = pd.DataFrame({'feature': ftr.columns, 'importance': rf_clf.feature_importances_})
imp = imp.sort_values('importance').set_index('feature')
imp.plot(kind='barh')
<matplotlib.axes._subplots.AxesSubplot at 0x1a4952bcd08>

```   


```python
#이상치 제거
for i in range(len(fea)):
    trn=trn[trn[fea[i]]>np.min(tst[fea[i]], axis=0)]
    trn=trn[trn[fea[i]]<np.max(tst[fea[i]], axis=0)]

for df in [trn,tst]:    

    for c in ugriz[1:]:
        div_col=f'div_u_{c}'
        df[div_col]=df.u/df[c]

    for c in ugriz:
        div_col=f'div_redshift_{c}'
        mul_col=f'mul_redshift_{c}'
        df[div_col]=df.redshift/df[c]
        df[mul_col]=df.redshift*df[c]
        
    df['div_abs_redshift_mean_u']=abs(df.redshift-np.mean(df.u))
    df['redshift_2']=df.redshift**2
    df['redshift_4']=df.redshift**4

    df['log_redshift_2']=np.log(df.redshift**2)
    df['log_redshift_4']=np.log(df.redshift**4)
    df['redshift_med_2']=(df.redshift-np.median(df.redshift))**2
    df['redshift_med_3']=(df.redshift-np.median(df.redshift))**3
    df["redshift_mean_2"]=(df.redshift-np.mean(df.redshift))**2

    df['asinh_mu'] = -2.5/np.log(10)*(np.arcsinh(df.u/24.63/(2.8e-10))-22.689378693319245)
    df['asinh_mg'] = -2.5/np.log(10)*(np.arcsinh(df.g/25.11/(1.8e-10))-23.131211445598282)
    df['asinh_mr'] = -2.5/np.log(10)*(np.arcsinh(df.r/24.80/(2.4e-10))-22.843529373146502)
    df['asinh_mi'] = -2.5/np.log(10)*(np.arcsinh(df.i/24.36/(3.6e-10))-22.43806426503834)
    df['asinh_mz'] = -2.5/np.log(10)*(np.arcsinh(df.z/22.83/(1.48e-09))-21.024370929730330)

    df['redshift_u_g'] = df.redshift-df.u-df.g
    df['redshift_u_r'] = df.redshift-df.u-df.r
    df['redshift_g_r'] = df.redshift-df.g-df.r
    df['redshift_4_log_u_2'] = df.redshift**4+np.log(df.u**2)

    for c in ugriz:
        div_col=f'div_redshift_2_{c}'
        df[div_col]=df.redshift**2/df[c]
    
    for c in asinh:
        mul_col=f'mul_{c}_redshift'
        div_col=f'div_{c}_redshift'
        df[mul_col]=df[c]*df.redshift
        df[div_col]=df[c]/df.redshift
        
    for f in [ugriz,dered,airmass]:
        for c1,c2 in itertools.combinations(f,2):
            dif_col=f'diff_{c1}_{c2}'
            df[dif_col]=df[c1]-df[c2]

    # 도메인 정보를 바탕으로 변수 추가
    # 출처 : http://classic.sdss.org/dr6/algorithms/sdssUBVRITransform.html
    df['Q_U_B']=0.75*(df.u-df.g)-0.81
    df['Q_B_V']=0.62*(df.g-df.r)+0.15
    df['Q_V_R']=0.38*(df.r-df.i)-0.2
    df['Q_Rc_IC']=0.72*(df.r-df.i)-0.27
    df['Q_B']=df.g+0.17*(df.u-df.g)+0.11
    df['Q_V']=df.g-0.52*(df.g-df.r)-0.03
    df['S_U_B']=0.77*(df.u-df.g)-0.88
    df['S_B_V']=0.90*(df.g-df.r)+0.21
    df['S_V_R']=0.96*(df.r-df.i)+ 0.21
    df['S_Rc_IC']=1.02*(df.r-df.i)+0.21
    df['S_B']=df.g+0.33*(df.g-df.r)+0.20
    df['S_V']=df.g -0.58*(df.g-df.r)- 0.01
    df['l-color'] = (-0.436*df['u']) + (1.129*df['g']) - (0.119*df['r']) - (0.574*df['i']) + (0.1984)
    df['s-color'] = (-0.249*df['u']) + (0.794*df['g']) - (0.555*df['r']) + (0.234)
    
    df['domain1'] = 0.7*(df.g-df.r) + 1.2*((df.r-df.i) - 0.177) 
    df['domain2'] = (df.r-df.i) - (df.g-df.r)/4 - 0.177 
    df['domain3'] = 0.449 - (df.g-df.r)/6  
    df['domain4'] = 1.296 + 0.25*(df.r-df.i)  
    df['domain5'] =  (df.r-df.i) - (df.g-df.r)/4 - 0.18
```

```python
    del df['u']
```

