
from typing import Tuple,List
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from data_load import boston_dt
import warnings
from sklearn.exceptions import ConvergenceWarning
import joblib
import os

def sklearn_train_val_test_split(
    boston_dt: pd.DataFrame,
    target: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Split Boston Housing into train / validation / test sets.
    """
    X = boston_dt.drop(columns=[target])
    y = boston_dt[target]

    # Step 1: test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Step 2: validation split
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio,
        random_state=random_state, shuffle=True
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
X_train, X_val, X_test, y_train, y_val, y_test = sklearn_train_val_test_split(boston_dt, target="MEDV")

len(X_train)

log_features = ["CRIM", "ZN", "NOX", "DIS", "LSTAT"]
yeo_johnson_features = ["B"]
none_features = ["RM"]
binary_features = ["CHAS"]
onehot_features = ["RAD"]
optional_binary = ["ZN"]
# transformations
log_transformer = FunctionTransformer(np.log1p, validate=True)
yeo_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
robust_scaler = RobustScaler()
onehot_encoder = OneHotEncoder(sparse_output=False, drop="if_binary")

preprocessor = ColumnTransformer(
    transformers=[
        ("log", Pipeline([("log", log_transformer), ("scale", robust_scaler)]), log_features),
        ("yeo", Pipeline([("yj", yeo_transformer), ("scale", robust_scaler)]), yeo_johnson_features),
        ("none_scale", Pipeline([("scale", robust_scaler)]), none_features),
        ("binary", "passthrough", binary_features),
        ("onehot", onehot_encoder, onehot_features),
        ("ZN_gt0", FunctionTransformer(lambda X: (X>0).astype(int), validate=True), optional_binary)
    ]
)
def style_df(df: pd.DataFrame, caption: str, width: str = "100%"):
    """
    table style for notebook inspection.
    """
    styles = [{"selector": "th","props": [ ("background-color", "#080808"), ("color", "white"),("font-weight", "bold"),("text-align", "center"),]},
        { "selector": "td","props": [("padding", "6px"), ("text-align", "center")],},]
    return (
        df.style.set_table_styles(styles).set_caption(caption).set_properties(**{"border": "1px solid #ddd"}) .set_table_attributes(f'style="width:{width};border-collapse:collapse"')
    )
def transform_data(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    X_transformed = preprocessor.fit_transform(X)
    feature_names = build_feature_names(preprocessor=preprocessor,log_features=log_features, yeo_features=yeo_johnson_features, none_features=none_features, binary_features=binary_features,
        onehot_features=onehot_features,
        optional_binary=optional_binary)
    return pd.DataFrame(X_transformed, columns=feature_names)
def build_feature_names(preprocessor: ColumnTransformer, log_features: List[str],yeo_features: List[str],none_features: List[str],binary_features: List[str],onehot_features: List[str],
                         optional_binary: List[str],
) -> List[str]:
    log_names = [f"{c}_log" for c in log_features]
    yj_names = [f"{c}_yj" for c in yeo_features]
    none_names = none_features.copy()
    binary_names = binary_features.copy()
    onehot_names = (
        preprocessor.named_transformers_["onehot"]
        .get_feature_names_out(onehot_features)
        .tolist()
    )
    optional_names = [f"{c}_gt0" for c in optional_binary]
    return (log_names+ yj_names+ none_names+ binary_names+ onehot_names+ optional_names)
def transformer_summary_table(log_features: List[str],yeo_features: List[str], none_features: List[str], binary_features: List[str],onehot_names: List[str],optional_binary: List[str],
) -> pd.DataFrame:
    """
    Table mapping each output feature to its applied transformation.
    """
    rows = []
    rows += [{"feature": f"{c}_log", "transformation": "log1p → RobustScaler"} for c in log_features]
    rows += [{"feature": f"{c}_yj", "transformation": "Yeo-Johnson → RobustScaler"} for c in yeo_features]
    rows += [{"feature": c, "transformation": "RobustScaler"} for c in none_features]
    rows += [{"feature": c, "transformation": "passthrough (binary)"} for c in binary_features]
    rows += [{"feature": c, "transformation": "OneHotEncoder"} for c in onehot_names]
    rows += [{"feature": f"{c}_gt0", "transformation": "(X > 0) → int"} for c in optional_binary]
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)

def onehot_result_table(
    X_transformed: pd.DataFrame, onehot_names: List[str], n_rows: int = 10
) -> pd.DataFrame:
    """
    OneHotEncoder output values.
    """
    return X_transformed[onehot_names].head(n_rows)
def scaled_result_table(X_transformed: pd.DataFrame,log_features: List[str],yeo_features: List[str],none_features: List[str],n_rows: int = 10,) -> pd.DataFrame:
    """
    scaled numeric feature values.
    """
    cols = (
        [f"{c}_log" for c in log_features]
        + [f"{c}_yj" for c in yeo_features]
        + none_features
    )
    return X_transformed[cols].head(n_rows)

X_train_transformed = transform_data(preprocessor, X_train)
X_test_transformed = transform_data(preprocessor,X_test)
X_val_transformed = transform_data(preprocessor,X_val)
onehot_names = (preprocessor.named_transformers_["onehot"].get_feature_names_out(onehot_features).tolist()

)
display(
    style_df(transformer_summary_table(log_features,yeo_johnson_features,none_features,binary_features,onehot_names,optional_binary,),
        caption="Feature → Transformer Mapping",
        width="70%",
    )
)

display(
    style_df(
        onehot_result_table(X_train_transformed, onehot_names),
        caption="OneHotEncoder Output (First 10 Rows)",
        width="80%",
    )
)

display(
    style_df(
        scaled_result_table(X_train_transformed,log_features,yeo_johnson_features,none_features),
        caption="Scaled Features (First 10 Rows)",
        width="80%",
    )
)

X_train.head()

def build_linear_pipeline() -> Pipeline:
    return Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])

def build_tree_pipeline() -> Pipeline:
    return Pipeline([
("model", DecisionTreeRegressor(max_depth=5, random_state=42))])

def build_mlp_pipeline() -> Pipeline:
    return Pipeline([("preprocess", preprocessor),
                     ("model", MLPRegressor(hidden_layer_sizes=(32,),activation="relu",solver="adam",max_iter=1000,early_stopping=True,n_iter_no_change=50,random_state=42))])

def evaluate_model(model: Pipeline,
                   X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   name: str) -> None:
    """
    Fit model and report RMSE and R2 on validation set.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)

    print(f"{name}")
    print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")
    print("-"*30)

# Pipelines
linear_pipe = build_linear_pipeline()
tree_pipe = build_tree_pipeline()
mlp_pipe = build_mlp_pipeline()
#Evaluate
warnings.filterwarnings("ignore", category=ConvergenceWarning)
evaluate_model(linear_pipe, X_train, y_train, X_val, y_val, "Linear Regression")
evaluate_model(tree_pipe, X_train, y_train, X_val, y_val, "Decision Tree")
evaluate_model(mlp_pipe, X_train, y_train, X_val, y_val, "Naive MLP")