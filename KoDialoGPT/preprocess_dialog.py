from lib_dialog import pd

# 데이터 불러오는 함수


def preprocess():

    train_data = pd.read_csv('new_train.csv')  # 파일 경로
    validate_data = pd.read_csv('new_test.csv')  # 파일 경로

    train_contexted = []

    for i in range(len(train_data)):
        row = []
        row.append(train_data.loc[i][1])
        row.append(train_data.loc[i][0])
        train_contexted.append(row)

    validate_contexted = []

    for i in range(len(validate_data)):
        row = []
        row.append(validate_data.loc[i][1])
        row.append(validate_data.loc[i][0])
        validate_contexted.append(row)

    columns = ['response', 'context']
    columns = columns + ['context/'+str(i) for i in range(0)]

    len(train_contexted)
    trn_df = pd.DataFrame.from_records(train_contexted, columns=columns)

    len(validate_contexted)
    val_df = pd.DataFrame.from_records(validate_contexted, columns=columns)

    return trn_df, val_df
