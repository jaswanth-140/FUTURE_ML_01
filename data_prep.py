import pandas as pd


def load_and_prep_data(filepath):
    # Load and filter
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df[(df['store'] == 1) & (df['item'] == 1)].copy()
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Rows loaded : {len(df)}")
    print(f"Date range  : {df['date'].min().date()} -> {df['date'].max().date()}")

    # Feature Engineering (Using 0-based year index)
    base_year = df['date'].dt.year.min()
    df['year_idx'] = df['date'].dt.year - base_year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    features = ['year_idx', 'month', 'day_of_week', 'is_weekend']
    target = 'sales'

    n_zeros = (df[target] == 0).sum()
    if n_zeros > 0:
        print(f"  [Warning] {n_zeros} zero-sales rows detected — MAPE will skip them.")

    # Chronological Split
    train_data = df[df['date'].dt.year <= 2016].copy()
    test_data = df[df['date'].dt.year == 2017].copy()

    X_train = train_data[features].values
    y_train = train_data[target].values
    X_test = test_data[features].values
    y_test = test_data[target].values

    print(f"Train : {len(X_train)} samples | Test : {len(X_test)} samples\n")

    return X_train, X_test, y_train, y_test, test_data, features, base_year