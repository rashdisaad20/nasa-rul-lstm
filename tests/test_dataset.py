from src.dataset import load_and_prepare_data


def test_load_and_prepare_data():
    X_train, y_train, X_test, y_test, scaler, feature_cols = load_and_prepare_data(
        'data/train_FD001.txt',
        'data/test_FD001.txt',
        'data/RUL_FD001.txt',
        sequence_length=30,
    )

    assert X_train.ndim == 3
    assert y_train.ndim == 1
    assert X_test.ndim == 3
    assert y_test.ndim == 1
    assert X_train.shape[1] == 30
    assert X_train.shape[2] == len(feature_cols)
    assert X_test.shape[1] == 30
    assert X_test.shape[2] == len(feature_cols)
    assert scaler is not None
    assert isinstance(feature_cols, list)
