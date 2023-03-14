from lib_chat_kobart import train_test_split

def preprocessing(df_db):
    
    # label 컬럼 없으면 0으로 생성
    if 'label' not in df_db.columns:
        df_db['label'] = 0

    # 컬럼 정렬
    df_db = df_db[['Q', 'A', 'label']]  
    
    # 중복, 결측치 제거
    df_db = df_db.drop_duplicates()     # 중복 제거
    df_db = df_db.dropna(axis=0)        # 결측치 제거
 
    # 데이터셋 분리
    df_train, df_valid = train_test_split(df_db, test_size=0.2, random_state=42, shuffle=True)
    
    return df_train, df_valid