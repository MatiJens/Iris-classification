import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data_from_csv(csv_path):
    """
    Load data from csv file, separating features and labels.
    Creating train and test sets and normalizing them.

    :param csv_path: Path to file
    :return X_train, X_test: Train and test sets as PyTorch tensors.
    """
    try:
        Iris_df = pd.read_csv(csv_path)
        Iris_df['Species'] = Iris_df['Species'].astype('category').cat.codes
    except FileNotFoundError:
        print(f"{csv_path} not found.")
        exit()

    # Separation features and labels
    X_numpy = Iris_df.iloc[:, 1:5].values
    Y_numpy = Iris_df.iloc[:, 5].values

    # Conversion Numpy series to PyTorch tensors
    X_tensor = torch.from_numpy(X_numpy).float()
    Y_tensor = torch.from_numpy(Y_numpy).long()

    # Creating train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=42,
                                                        stratify=Y_tensor)
    # Creating standard scaler to normalize sets
    scaler = StandardScaler()

    # Transforming sets to numpy to normalize them
    X_train_np = scaler.fit_transform(X_train.numpy())
    X_test_np = scaler.transform(X_test.numpy())

    # Transforming normalized sets to tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)

    return X_train, X_test