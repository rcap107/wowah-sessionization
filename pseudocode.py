import polars as pl


def make_user_month():
    # char_month = outer product unique month x unique character
    # data = df drop duplicates([user, month]) add column 'has played' = True
    # char_month left join data on char_month.month = data.month and char_month.char = data.char
    # fillna (has played = False)
    # return joined['month', 'char', 'has played']
    pass

class Splitter:
    def split(self, user_month, has_played):
        for split_point in range(min date, max date, 1 month):
            train_idx = everything before split_point
            test_idx = everything after split_point
            yield train_idx, test_idx

def filter_past(X, historical_data):
    assert X['month'] is constant
    assert X['cid'] is unique
    return historical_data.filter(everything before X['date'][0] minus 1 month)

def make_data_op():
    user_month_has_played = skrub.var('query')
    X = user_month_has_played['user', 'month'].skb.mark_as_X(cv=Splitter())
    y = user_month_has_played['has_played'].skb.mark_as_y()
    historical_data_file = skrub.var('historical_data_file')
    historical_data = load(historical_data_file)
    kept_historical_data = filter_past(X, historical_data)
    features = add_features(X, kept_historical_data)
    data_op = features.skb.apply(HGB(), y=y)
    return data_op


def cross_validate():
    df = make_user_month()
    make_data_op().skb.cross_validate({'query': df})